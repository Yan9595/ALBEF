from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertLMHeadModel
from transformers import OPTForCausalLM, T5ForConditionalGeneration

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ALBEF_BLIP2(nn.Module):
    def __init__(self, 
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=6, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        config_encoder = BertConfig.from_json_file(config['bert_config'])   
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)  
            
        # self.text_decoder = OPTForCausalLM.from_pretrained(text_decoder)
        self.text_decoder = T5ForConditionalGeneration.from_pretrained(text_decoder)
        # self.text_decoder.resize_token_embeddings(config.vocab_size)

    def forward(self, image, question_bert, question_opt, k = None, answer = None, train = True):
        # Encode images
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        # Encode text
        question_output = self.text_encoder(question_bert.input_ids, 
                                                attention_mask = question_bert.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True)
        
        question_output = question_output.last_hidden_state
        question_output = question_output[:, 0, :].unsqueeze(1)

        llm_embeds = self.text_decoder.get_input_embeddings()(question_opt.input_ids)
        llm_embeds = torch.cat([question_output, llm_embeds], dim=1)
        llm_atts = torch.ones((llm_embeds.size()[0], 1), dtype=torch.long).to(llm_embeds.device)
        llm_atts = torch.concat([llm_atts, question_opt.attention_mask.to(llm_embeds.device)], dim=1)
        
        # generate_kwargs["max_length"] = (generate_kwargs.get("max_length", 20) 
        #                                  + question_output.shape[1] - 1)
        # generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + question_output.shape[1]
        if train:
            # If not already a tensor
            k_tensor = torch.tensor(k, device=llm_embeds.device)

            # Repeat embeddings and atts
            llm_embeds = llm_embeds.repeat_interleave(k_tensor, dim=0)
            llm_atts = llm_atts.repeat_interleave(k_tensor, dim=0)
            encoder_outputs = self.text_decoder.encoder(
                inputs_embeds=llm_embeds,
                attention_mask=llm_atts,
                return_dict=True,
            )
            outputs = self.text_decoder(
                encoder_outputs=encoder_outputs,
                labels = answer
            )
            loss = outputs.loss
            return loss
        
        else:
            # outputs = self.text_decoder.generate(
            #     inputs_embeds=llm_embeds,
            #     attention_mask=llm_atts,
            #     do_sample=False,
            #     max_length=30,
            #     min_length=1,
            #     num_beams=1,
            #     early_stopping=True
            # )
            question_output = self.text_decoder.encoder(
                inputs_embeds=llm_embeds,
                attention_mask=llm_atts,
                return_dict=True,
            )
            
            topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, llm_atts, 
                                                    answer.input_ids, answer.attention_mask, k)
            return topk_ids, topk_probs

        
            

        # outputs = self.text_decoder(inputs_embeds = llm_embeds, 
        #                             attention_mask = llm_atts,
        #                             use_cache=True,
        #                             return_dict=True)
        
        # next_token_logits = outputs.logits[:, -1, :]
        # next_token = torch.argmax(next_token_logits, dim=-1)
        # output = next_token.clone()

        # while next_token != self.tokenizer.eos_token_id:
        #     past_key_values = outputs.past_key_values
        #     llm_atts = torch.cat([llm_atts, torch.ones((llm_embeds.size()[0], 1), dtype=torch.long).to(llm_embeds.device)], dim=1)
        #     next_embed = self.text_decoder.get_input_embeddings()(next_token)
        #     outputs = self.text_decoder(inputs_embeds = next_embed, 
        #                                 attention_mask = llm_atts,
        #                                 use_cache=True,
        #                                 return_dict=True,
        #                                 past_key_values=past_key_values)
        #     next_token_logits = outputs.logits[:, -1, :]
        #     next_token = torch.argmax(next_token_logits, dim=-1)

        #     output = torch.cat([output, next_token], dim=1)
        
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        num_ques = question_states.size(0)
        start_ids = (torch.tensor(self.text_decoder.config.decoder_start_token_id, dtype = torch.long, 
                    device = self.text_decoder.device).repeat(num_ques,1)) # bos token

        
        start_output = self.text_decoder.decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         ) 
                
        logits = start_output.last_hidden_state[:,0,:] # first token's logit
        logits = self.text_decoder.lm_head(logits)
        

        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,0]
        
        
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  
        
        
        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        start_ids = (torch.ones((num_ques*k,1), dtype=torch.long).
                    to(input_ids.device) * self.text_decoder.config.decoder_start_token_id)
        input_ids = torch.concat([start_ids, input_ids], dim=1)
        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
        input_atts = torch.concat([torch.ones((input_ids.size(0),1), dtype=torch.long).to(input_ids.device), input_atts], dim=1)

        input_ids = input_ids[:, :-1].contiguous()
        input_atts = input_atts[:, :-1].contiguous()
        targets_ids = targets_ids[:, 1:].contiguous()

        output = self.text_decoder(decoder_input_ids = input_ids, 
                                   decoder_attention_mask = input_atts, 
                                   encoder_outputs = [question_states],
                                   attention_mask = question_atts,     
                                   return_dict = True, 
                                   )                 

        logits = output.logits
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        logits = logits.view(-1, logits.size(-1))
        targets_ids = targets_ids.view(-1)
        answer_loss = loss_fct(logits, targets_ids)
        answer_loss = answer_loss.view(logits.size(0),-1).sum(1)
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs
    
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))
            



