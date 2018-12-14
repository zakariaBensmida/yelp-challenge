library(jsonlite)
library("keras")
untar("~/yelp_dataset.tar",files=NULL)
data<-jsonlite::stream_in(textConnection(readLines("~/yelp_academic_dataset_review.json", n=100000)),verbose=F)
yelp20018<-data
labels <- yelp20018$stars
for (i in 1:100000){
  
if (labels[i]<=3 ){ labels[i]<-0}
else if (labels[i]>=4 ){labels[i]<-1}
}

tokenizer <- text_tokenizer(num_words = 20000) %>%  fit_text_tokenizer(yelp20018$text)
sequences <- texts_to_sequences(tokenizer, yelp20018$text)
data1 <- pad_sequences(sequences, maxlen = 300)



model <- keras_model_sequential() %>%  
  layer_embedding(input_dim = 20000, output_dim = 32) %>% 
  layer_dropout(rate = 0.5)%>%
  layer_lstm(units = 32) %>%  
  layer_dropout(rate = 0.5)%>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(  
  optimizer = "rmsprop", 
  loss = "binary_crossentropy",
  metrics = c("acc") 
  )
history <- model %>% fit(  
  data1, labels,  
        epochs = 4,  
        batch_size = 128,  
        validation_split = 0.5 
)


