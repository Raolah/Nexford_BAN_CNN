# Load necessary libraries
library(keras)
library(ggplot2)

# Load the fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
train_images <- fashion_mnist$train$x
train_labels <- fashion_mnist$train$y
test_images <- fashion_mnist$test$x
test_labels <- fashion_mnist$test$y

# Normalize the images to a range of 0 to 1 by dividing by 255
train_images <- train_images / 255
test_images <- test_images / 255

# Reshape the data by adding a channel dimension to the images
train_images <- array_reshape(train_images, c(dim(train_images)[1], 28, 28, 1))
test_images <- array_reshape(test_images, c(dim(test_images)[1], 28, 28, 1))

# Build the CNN Model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit(
  train_images, train_labels,
  epochs = 10,
  validation_data = list(test_images, test_labels)
)

# Evaluate the model for prediction
test_eval <- model %>% evaluate(test_images, test_labels, verbose = 2)
cat(sprintf("Test accuracy: %.4f\n", test_eval$accuracy))

# Make predictions
predictions <- model %>% predict(test_images)

# Display the prediction for the two images
plot_image <- function(i, predictions_array, true_label, img) {
  predictions_array <- predictions_array[i, ]
  true_label <- true_label[i]
  img <- img[i, , , drop = FALSE]
  img <- img[,,1] # Select the first channel
  par(mfrow=c(1,1))
  plot(as.raster(img), axes=FALSE, col=gray.colors(256))
  predicted_label <- which.max(predictions_array) - 1
  color <- ifelse(predicted_label == true_label, 'blue', 'red')
  title(main = sprintf("%d (%d)", predicted_label, true_label), col.main=color)
}

plot_value_array <- function(i, predictions_array, true_label) {
  predictions_array <- predictions_array[i, ]
  true_label <- true_label[i]
  barplot(predictions_array, col="#777777", ylim=c(0, 1), names.arg=0:9)
  predicted_label <- which.max(predictions_array) - 1
  rect(predicted_label + 0.5, 0, predicted_label + 1.5, predictions_array[predicted_label + 1], col='red')
  rect(true_label + 0.5, 0, true_label + 1.5, predictions_array[true_label + 1], col='blue')
}

num_rows <- 1
num_cols <- 2
num_images <- num_rows * num_cols
par(mfrow=c(num_rows, 2 * num_cols))
for (i in 0:(num_images - 1)) {
  plot_image(i + 1, predictions, test_labels, test_images)
  plot_value_array(i + 1, predictions, test_labels)
}

