from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf


def transformers(model_name, context, question):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
    inputs = tokenizer(question, context, return_tensors='tf')
    outputs = model(inputs)
    answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    answer_end = tf.argmax(outputs.end_logits, axis=1).numpy()[0] + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    
    print(f"\nContext: {context}\nQuestion: {question}\nAnswer: {answer}")