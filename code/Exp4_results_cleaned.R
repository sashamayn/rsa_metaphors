### Importing packages ###
library(lsr)
library(tidyr)
library(reshape)
library(ggplot2)
require(gridExtra)
library(dplyr)

model_data <- read.csv("OUTPUTOFMODEL.csv")
means_data <- read.csv("data/predictions.csv")
priors <- read.csv('data/priors.csv')

both <- merge(means_data,model_data,by=c("ChosenFeature","Adj","Animal"))

# we only consider averages of more than 10 ratings
TenOrMoreRatings <- subset(both,both$NumRatings>=10)
TenOrMoreRatings$Label <- paste(TenOrMoreRatings$ChosenFeature,"",TenOrMoreRatings$Animal)

specific <- subset(TenOrMoreRatings,TenOrMoreRatings$Condition=='specific')
specific$Typicality <- factor(specific$Typicality)

### Model Priors vs Probabilities ###
priors_long <- melt(priors, id.vars = c("X"), variable.name = "Rating")
colnames(priors_long)[1] <- "ChosenFeature"
colnames(priors_long)[2] <- "Animal"

predictions_w_priors <- merge(model_data,priors_long,by=c("Animal","ChosenFeature"))
predictions_w_priors <- subset(predictions_w_priors, predictions_w_priors$Adj != "na")
predictions_w_priors$ChosenFeature <- factor(predictions_w_priors$ChosenFeature)
predictions_w_priors$Adj <- factor(predictions_w_priors$Adj)
predictions_w_priors <- subset(predictions_w_priors, ChosenFeature==Adj)
predictions_w_priors$Label <- paste(predictions_w_priors$Adj,predictions_w_priors$Animal)


priors_df <- predictions_w_priors %>% select(Animal,ChosenFeature,value)
colnames(priors_df)[3] <- 'TypPrior'
total <- merge(TenOrMoreRatings, priors_df,by=c("ChosenFeature","Animal"))

#Human certainty ratings plotted against typicality priors
p1 <- ggplot(data=total, aes(x=TypPrior,y=FeatureCertaintyMean,color=Condition,shape=Typicality)) + geom_point(size = 2)+
  geom_text(aes(label=Label),hjust=0.5, vjust=-0.7) +
  geom_smooth(method='lm',se = FALSE) +
  ggtitle("Human certainty against typicality priors")

#Model probabilities plotted against typicality priors
p2 <- ggplot(data=total, aes(x=TypPrior,y=ModelProb,color=Condition,shape=Typicality)) + geom_point(size = 2)+
  geom_text(aes(label=Label),hjust=0.5, vjust=-0.7) +
  geom_smooth(method='lm',se = FALSE) +
  ggtitle("Model probabilities against typicality priors")

### Human vs Model ####

# Feature certainty plotted against model probabilities
corrPlot <- ggplot(data=TenOrMoreRatings, aes(x=ModelProb,y=FeatureCertaintyMean,color=Condition,shape=Typicality)) + geom_point(size = 2)+
  geom_text(aes(label=Label),hjust=0.5, vjust=-0.7) +
  geom_smooth(method='lm',se = FALSE) +
  ggtitle("Model probabilities vs human certainty ratings")

# Feature degree plotted against model probabilities
degreePlot <- ggplot(data=TenOrMoreRatings, aes(x=FeatureDegreeMean,y=ModelProb,color=Condition,shape=Typicality)) + geom_point(size = 2)+
  geom_text(aes(label=Label),hjust=0.5, vjust=-0.7) +
  geom_smooth(method='lm',se = FALSE) +
  ggtitle("Human degree against model predictions, corr=0.67")

# Pearson correlation between human certainty ratings and model probabilities
cor(TenOrMoreRatings$FeatureCertaintyMean, TenOrMoreRatings$ModelProb,  method = "pearson", use = "complete.obs") 

# Pearson correlation with just the specific data
cor(specific$FeatureCertaintyMean, specific$ModelProb,  method = "pearson", use = "complete.obs")

# Correlation between human degree ratings and model probabilities
cor(TenOrMoreRatings$FeatureDegreeMean, TenOrMoreRatings$ModelProb,  method = "pearson", use = "complete.obs")

# To what extent feature certainty ratings are explained by model probabilities
rm3 <- lm(TenOrMoreRatings$FeatureCertaintyMean~TenOrMoreRatings$ModelProb)


#######################


#### Human Data Analyses ###

# For the specific conditions, to what extend human degree ratings are predicted by typicality priors. 
# Highly (R-squared = 0.79, all ps<0.001).
rm <- lm(FeatureDegreeMean~Typicality, data=specific)

#Barplot of average degree ratings by typicality
plot(FeatureDegreeMean~Typicality,data=specific, main='average degree ratings by typicality')

# To what extent typicality priors are predictive of people's certainty ratings.
# R-squared = 0.26, all ps < 0.001.
rm2 <- lm(FeatureCertaintyMean~Typicality, data=specific)

# Barplot of certainty rating by typicality condition. 
# Highest certainty for high typicality, followed by low, followed by avg.
plot(FeatureCertaintyMean~Typicality,data=specific, main='certainty rating by typicality')

# For the vague condition, a plot of spread of certainty ratings
long_small <- melt(priors, id.vars = c("X"), variable.name = "Rating")
colnames(long_small)[1] <- "ChosenFeature"
colnames(long_small)[2] <- "Animal"
colnames(long_small)[3] <- "ChosenFeatureTypPrior"

vague <- subset(both, both$Condition=='vague')
vague_w_priors <- merge(vague,long_small,by=c("ChosenFeature","Animal"))
vague_w_priors$NumRatings <- factor(vague_w_priors$NumRatings)

scale_colour_gradient(high = "#132B43", low = "#56B1F7")

ggplot(data=vague_w_priors, aes(x=ChosenFeatureTypPrior,y=FeatureDegreeMean,color=FeatureCertaintyMean)) +
  geom_text(aes(label=NumRatings),hjust=0.5, vjust=-0.7, size=5) + scale_colour_gradient(high = "#132B43", low = "#56B1F7") +
ggtitle("Typ priors vs feature certainty in the vague condition")


#######################

