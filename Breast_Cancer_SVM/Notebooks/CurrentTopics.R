#First, installing necessary packages and libraries for pre-processing
install.packages(c("tidyverse", "e1071", "caret", "corrplot"))
# Load tidyverse
library(tidyverse)
library(e1071)
library(caret)
library(corrplot)
#====================================================================================================
#**Loading the data from Google Drive: Please ensure that the data is saved as BreastCancerData.csv in your Google Drive**

# Mount Google Drive
library(googledrive)
library(readr)

drive_deauth()  # avoid auth issues in shared work
drive_auth()    # you may be prompted to grant access

# Make sure your csv file is exactly under the My Drive, not in any other subfolder!!!
# Use file picker (or set filename manually if known)
file <- drive_get("BreastCancerData.csv")  # change if your file name is different
drive_download(file, path = "BreastCancerData.csv", overwrite = TRUE)

# Read the downloaded file
data <- read_csv("BreastCancerData.csv")

#====================================================================================================

#**Overall view of the dataset**

# Number of rows and columns
cat("Number of rows:", nrow(data), "\n")
cat("Number of columns:", ncol(data), "\n")

# Class distribution
cat("Diagnosis class distribution:\n")
print(table(data$diagnosis))

# Quick summary of all variables
cat("Summary of numeric features:\n")
print(summary(data %>% select(where(is.numeric))))

# Check column names
cat("Column names:\n")
print(colnames(data))

#====================================================================================================
#The dataset initially includes an id column used solely for identification purposes and an unnamed column (...33) 
#that contains only missing values. Since these features provide no informative value for classification, they will be removed.

# Drop unnecessary columns
data <- data %>% select(-id, -`...33`)

# Convert diagnosis to factor
data$diagnosis <- as.factor(data$diagnosis)

# Confirm structure
str(data)

#====================================================================================================

#**Distribution of Target Variable, Diagnosis. B = Benign, M = Malignant**
ggplot(data, aes(x = diagnosis, fill = diagnosis)) +
  geom_bar() +
  scale_fill_manual(values = c("B" = "steelblue", "M" = "firebrick")) +
  labs(
    title = "Distribution of Diagnosis Classes",
    x = "Diagnosis (B = Benign, M = Malignant)",
    y = "Count"
  ) +
  theme_minimal()

#====================================================================================================

#**Checking for Skewness in Features**

data_long <- data %>%
  pivot_longer(
    cols = -diagnosis,
    names_to = "Feature",
    values_to = "Value"
  )
#Plotting distributions of features
ggplot(data_long, aes(x = Value, fill = diagnosis)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  facet_wrap(~ Feature, scales = "free", ncol = 4) +
  scale_fill_manual(values = c("B" = "steelblue", "M" = "firebrick")) +
  labs(
    title = "Feature Distributions by Diagnosis",
    x = "Value",
    y = "Count"
  ) +
  theme_minimal()

  # Remove the diagnosis column so we’re only checking numeric features
numeric_data <- data %>% select(-diagnosis)

# Calculate skewness for each feature
skew_values <- apply(numeric_data, 2, skewness)
skew_values_sorted <- sort(skew_values, decreasing = TRUE)

# Convert to data frame for better readability
skew_df <- data.frame(
  Feature = names(skew_values_sorted),
  Skewness = round(skew_values_sorted, 3)
)

# Print top 10 most skewed features nicely
head(skew_df, 10)

# Define top 5 most skewed features
top_skewed_features <- c("area_se", "concavity_se", "fractal_dimension_se", "perimeter_se", "radius_se")

# Filter data_long to only include these features
top_skewed_data <- data_long %>%
  filter(Feature %in% top_skewed_features)

# Plot distributions
ggplot(top_skewed_data, aes(x = Value, fill = diagnosis)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  facet_wrap(~ Feature, scales = "free", ncol = 2) +
  scale_fill_manual(values = c("B" = "steelblue", "M" = "firebrick")) +
  labs(
    title = "Distributions of Top 5 Most Skewed Features",
    x = "Feature Value",
    y = "Count"
  ) +
  theme_minimal()

#====================================================================================================

#**Check for Missing Values**

missing_values <- data.frame(
  Feature = colnames(data),
  Missing_Values = colSums(is.na(data))
)

print(missing_values)

#====================================================================================================

#**Train/Test Split**

# Split data into training and test set to prevent data leakage

# Set seed for reproducibility
set.seed(123)

# Create stratified indices
train_index <- createDataPartition(data$diagnosis, p = 0.8, list = FALSE)

# Split the data
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# Check class balance
cat("Train set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")
table(train_data$diagnosis)
table(test_data$diagnosis)

#====================================================================================================

#**Normalizing Features**

# Create a pre-processing object (excluding the diagnosis column)
preproc <- preProcess(train_data %>% select(-diagnosis), method = c("center", "scale"))

# The test set is normalized using the training set's distribution
train_normalized <- train_data
train_normalized[, -which(names(train_normalized) == "diagnosis")] <- predict(preproc, train_data %>% select(-diagnosis))

test_normalized <- test_data
test_normalized[, -which(names(test_normalized) == "diagnosis")] <- predict(preproc, test_data %>% select(-diagnosis))

data_long_norm <- train_normalized %>%
  pivot_longer(
    cols = -diagnosis,
    names_to = "Feature",
    values_to = "Value"
  )

head(data_long_norm, 10)

#Plotting normalized feature distribution of training data
ggplot(data_long_norm, aes(x = Value, fill = diagnosis)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  facet_wrap(~ Feature, scales = "free", ncol = 4) +
  scale_fill_manual(values = c("B" = "steelblue", "M" = "firebrick")) +
  labs(
    title = "Normalized Feature Distributions of Training Data by Diagnosis",
    x = "Normalized Value",
    y = "Count"
  ) +
  theme_minimal()

#====================================================================================================
#**Check for Multicollinearity - Correlation amongst features**

# Drop the target variable (diagnosis) to focus on numeric features
numeric_data <- train_normalized %>% select(-diagnosis)

# Compute the correlation matrix
cor_matrix <- cor(numeric_data)

# Plot correlation heatmap
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.col = "black",
         tl.srt = 45,
         title = "Feature Correlation Heatmap",
         mar = c(0, 0, 2, 0))

#====================================================================================================
#**Features with correlation above 0.9. Optional to drop highly correlated features**

# Use caret's findCorrelation to identify features with correlation > 0.9

high_corr <- findCorrelation(cor_matrix, cutoff = 0.9, names = TRUE, verbose = TRUE)

# Convert the correlation matrix into a tidy dataframe of pairs
cor_pairs <- as.data.frame(as.table(cor_matrix)) %>%
  filter(Var1 != Var2) %>%                             # exclude self-correlations
  mutate(abs_corr = abs(Freq)) %>%
  filter(abs_corr > 0.9) %>%                           # filter for strong correlations
  arrange(desc(abs_corr)) %>%
  distinct(pmin(Var1, Var2), pmax(Var1, Var2), .keep_all = TRUE) %>%  # remove duplicate pairs
  select(Feature_1 = Var1, Feature_2 = Var2, Correlation = Freq)

# Display result
print(cor_pairs)

# Drop highly correlated features based on correlation pairs since SVM model is sensitive to multicollinearity
features_to_drop <- c(
  "perimeter_mean",
  "perimeter_worst",
  "area_mean",
  "area_worst",
  "perimeter_se",
  "area_se",
  "concave points_mean",
  "texture_mean"
)

train_normalized_reduced <- train_normalized %>% select(-all_of(features_to_drop))

# To understand feature importance

print("\nFeature Importance Analysis:")
diagnosis_numeric <- as.numeric(train_normalized_reduced$diagnosis) - 1  # Convert to 0/1
feature_importance <- data.frame(
  Feature = names(train_normalized_reduced)[-which(names(train_normalized_reduced) == "diagnosis")],
  Correlation = sapply(train_normalized_reduced[, -which(names(train_normalized_reduced) == "diagnosis")],
                      function(x) abs(cor(x, diagnosis_numeric)))
)

# Sort by absolute correlation
feature_importance <- feature_importance[order(-feature_importance$Correlation),]
print(feature_importance)

# Plot top 10 most important features
print("\nPlotting top 10 most important features...")
top_features <- head(feature_importance$Feature, 10)
data_long_top <- train_normalized_reduced %>%
  select(all_of(top_features), diagnosis) %>%
  pivot_longer(cols = -diagnosis, names_to = "Feature", values_to = "Value")

print(ggplot(data_long_top, aes(x = Value, fill = diagnosis)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ Feature, scales = "free", ncol = 2) +
  scale_fill_manual(values = c("B" = "steelblue", "M" = "firebrick")) +
  labs(
    title = "Distribution of Top 10 Most Important Features",
    x = "Value",
    y = "Density"
  ) +
  theme_minimal())

# Reshape to long format for plotting
data_long <- train_data %>%
  pivot_longer(cols = -diagnosis, names_to = "Feature", values_to = "Value")

# Plot boxplots for selected features
ggplot(
  data_long %>% filter(Feature %in% c("radius_mean", "area_mean", "concavity_mean", "smoothness_mean")),
  aes(x = diagnosis, y = Value, fill = diagnosis)
) +
  geom_boxplot(alpha = 0.6) +
  facet_wrap(~ Feature, scales = "free", ncol = 2) +
  scale_fill_manual(values = c("B" = "steelblue", "M" = "firebrick")) +
  labs(
    title = "Boxplots of Key Features by Diagnosis",
    x = "Diagnosis",
    y = "Value"
  ) +
  theme_minimal()

#====================================================================================================

#**Model: Support Vector Machine**
install.packages("kernlab")
install.packages("themis")
install.packages("fpc")  # For silhouette plot
library(fpc)
install.packages("cluster")
library(cluster)

# Load required packages
library(e1071)
library(caret)
library(pROC)
library(ggplot2)

print("Training model...")
model <- train(
  diagnosis ~ .,
  data = train_normalized_reduced,
  method = "svmRadial",
  metric = "ROC",
  preProcess = c("center", "scale"),
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
  tuneGrid = data.frame(C = 1, sigma = 0.1)
)

# Evaluate on test set
print("Model Performance on Test Set:")
test_predictions <- predict(model, test_normalized[, -which(names(test_normalized) == "diagnosis")])
test_probs <- predict(model, test_normalized[, -which(names(test_normalized) == "diagnosis")], type = "prob")

# Calculate test performance metrics
test_cm <- confusionMatrix(test_predictions, test_normalized$diagnosis)
test_auc <- roc(test_normalized$diagnosis, test_probs[,2])$auc

# Print test performance
cat("\nTest Accuracy:", round(test_cm$overall['Accuracy'], 4))
cat("\nTest AUC:", round(test_auc, 4))
cat("\nTest Confusion Matrix:\n")
print(test_cm$table)

#====================================================================================================
#**Visualization Techniques**

#1. ROC Curve

# Plot ROC curve for test set
roc_curve <- roc(test_normalized$diagnosis, test_probs[,2])
plot(roc_curve, main = "ROC Curve (FPR vs TPR)", col = "blue", legacy.axes = TRUE)

abline(a = 0, b = 1, lty = 2, col = "red")
text(0.7, 0.3, paste("AUC =", round(test_auc, 4)))
#====================================================================================================

#2 Silhouette Plot
# Get decision function for test set
svm_decision_scores <- predict(model, test_normalized[, -which(names(test_normalized) == "diagnosis")], type = "raw")
svm_decision_scores <- as.numeric(svm_decision_scores)

# Calculate Silhouette scores for test set
dist_matrix <- dist(test_normalized[, -which(names(test_normalized) == "diagnosis")])
sil_score <- silhouette(as.numeric(svm_decision_scores), dist_matrix)

# Plot the Silhouette plot
plot(sil_score, main = "Silhouette Plot for SVM Model", col = 2:3)
#====================================================================================================

#3. Classmap
print("\nCreating Classmap...")
# Get prediction probabilities
pred_probs <- predict(model, test_normalized[, -which(names(test_normalized) == "diagnosis")], type = "prob")

# Create separate data frames for each class
benign_data <- data.frame(
  prob = pred_probs[test_normalized$diagnosis == "B", "M"],
  fairness = seq(0, 1, length.out = sum(test_normalized$diagnosis == "B")),
  true_class = "B",
  predicted = test_predictions[test_normalized$diagnosis == "B"]
)

malignant_data <- data.frame(
  prob = pred_probs[test_normalized$diagnosis == "M", "B"],
  fairness = seq(0, 1, length.out = sum(test_normalized$diagnosis == "M")),
  true_class = "M",
  predicted = test_predictions[test_normalized$diagnosis == "M"]
)

# Plot classmaps
par(mfrow = c(1, 2))

# Plot for Benign cases
plot(benign_data$fairness, benign_data$prob,
     col = ifelse(benign_data$predicted == "B", "steelblue", "firebrick"),
     pch = 19,
     main = "Class Map of Benign Cases",
     xlab = "Farness from Given Class",
     ylab = "P[alternative class]",
     ylim = c(0, 1))
abline(v = 0.99, lty = 2)
grid()

# Plot for Malignant cases
plot(malignant_data$fairness, malignant_data$prob,
     col = ifelse(malignant_data$predicted == "M", "firebrick", "steelblue"),
     pch = 19,
     main = "Class Map of Malignant Cases",
     xlab = "Farness from Given Class",
     ylab = "P[alternative class]",
     ylim = c(0, 1))
abline(v = 0.99, lty = 2)
grid()

#====================================================================================================

#4. Stacked Plot
print("\nCreating Stackedplot...")
# Calculate confusion matrix proportions
cm <- confusionMatrix(test_predictions, test_normalized$diagnosis)
cm_table <- cm$table
cm_props <- prop.table(cm_table, margin = 2)  # Calculate proportions by column (given class)

# Create data frame for plotting
plot_data <- data.frame(
  given_class = rep(colnames(cm_table), each = nrow(cm_table)),
  predicted_class = rep(rownames(cm_table), times = ncol(cm_table)),
  proportion = as.vector(cm_props)
)

# Plot stacked bar plot
print(ggplot(plot_data, aes(x = given_class, y = proportion, fill = predicted_class)) +
  geom_bar(stat = "identity", position = "stack", color = "white", linewidth = 0.5) +
  scale_fill_manual(values = c("B" = "brown", "M" = "pink")) +
  labs(
    title = "Classification Results",
    x = "Given Class",
    y = "Proportion",
    fill = "Predicted Class"
  ) +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    plot.title = element_text(hjust = 0.5)
  ))

#====================================================================================================

#5. Quasi Residual Plot
install.packages("patchwork")  # only once
library(patchwork)
# Add predicted probabilities and actual class to test data
plot_data <- test_data  # Use original test data (non-normalized)
plot_data$prob_malignant <- test_probs$M  # Predicted probability of malignant
plot_data$prob_benign <- test_probs$B    # Predicted probability of benign
plot_data$actual_class <- test_data$diagnosis
plot_data$area_mean <- test_data$area_mean  # Use the original 'area_mean' column

# Identify misclassified points: Benign points with high probability of Malignant
plot_data$misclassified <- ifelse(plot_data$actual_class == "B" & plot_data$prob_malignant > 0.5, "Misclassified", "Correct")

# Filter by actual class
benign_data <- plot_data[plot_data$actual_class == "B", ]
malignant_data <- plot_data[plot_data$actual_class == "M", ]

# Create individual plots
plot_benign <- ggplot(benign_data, aes(x = area_mean, y = prob_malignant, color = misclassified)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", color = "#1f77b4", se = FALSE) +  # Change line color to red
  scale_color_manual(values = c("Correct" = "#1f77b4", "Misclassified" = "#ff0000")) +  # Misclassified points in red
  labs(
    title = "Quasi-Residual Plot (True Class: Benign)",
    x = "Area Mean",
    y = "Predicted Probability of Malignant"
  ) +
  ylim(0, 1) +
  theme_minimal()

plot_malignant <- ggplot(malignant_data, aes(x = area_mean, y = prob_benign)) +  # Use prob_benign for the malignant plot
  geom_point(alpha = 0.6, color = "#ff0000") +  # Malignant points in red
  geom_smooth(method = "loess", color = "#ff0000", se = FALSE) +  # Change line color to red
  labs(
    title = "Quasi-Residual Plot (True Class: Malignant)",
    x = "Area Mean",
    y = "Predicted Probability of Benign"  # Changed label to match the new probability being plotted
  ) +
  ylim(0, 1) +
  theme_minimal()

# Set larger default plot size
options(repr.plot.width = 14, repr.plot.height = 6)  # Only needed in notebooks or inline display environments

# Show the side-by-side plots
plot_benign + plot_malignant

