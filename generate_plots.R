# plot learning rate data
library(ggplot2)
library(dplyr)

setwd("E:/classes/fall_2020/CSC_6240_Math_Theory_of_Machine_Learning/exam_4_video/")

dat <- read.csv(file = '1578results.csv')

learning_rate_plot <- ggplot(data = dat %>% filter(LR < 1) %>% filter(LR > 0))+
  geom_point(aes(x = LR, y = TE_ACC, col = MOMENTUM))+
  scale_color_viridis_c()
learning_rate_plot


# vary learning rate and momentum, fix everything else at its defaults
# decay is close to zero, default 1e-07
# learning rate defaults to 0.001 (vary this)
# num_layers default is 3
# hidden_dim, batch_sz, epochs all default to 50
# Nesterov defaults to false
# momentum defaults to 0.9 (vary this)

dat2 <- dat %>% 
  # filter(LR < 1) %>%
  # filter(LR > 0) %>%
  filter(DECAY == 1e-07) %>%
  filter(NUM_LAYERS == 3) %>%
  filter(HIDDEN_DIM == 50) %>%
  filter(BATCH_SZ == 50) %>%
  filter(EPOCHS == 50) %>%
  filter(!NESTEROV)


# keep this for the presentation
learning_rate_plot <- ggplot(data = dat2 %>%
                               filter(LR < 0.002) %>%
                               filter(LR > 0))+
  # geom_point(aes(x = LR, y = TE_ACC, col = MOMENTUM))
  geom_point(aes(x = MOMENTUM, y = TE_ACC, col = LR))+
  geom_line(aes(x = MOMENTUM, y = TE_ACC, col = LR))+
  scale_color_viridis_c()+
  theme_bw()+
  labs(x = "Momentum",
       y = "Testing Accuracy",
       col = "Learning\nRate")
learning_rate_plot



# keep this for the presentation
agg_dat_2 <- as.data.frame(dat2 %>%
                             filter(LR <= 0.25) %>%
                             filter(LR > 0) %>%
                             group_by(LR, MOMENTUM) %>%
                             summarize(LR = LR,
                                       MOMENTUM = MOMENTUM,
                                       TR_ACC = mean(TR_ACC),
                                       TE_ACC = mean(TE_ACC)))


learning_rate_plot2 <- ggplot(data = agg_dat_2)+
  geom_point(aes(x = LR, y =TE_ACC))+
  geom_contour_filled(aes(x = LR, y = MOMENTUM, z = TE_ACC, col = TE_ACC))+
  theme_bw()+
  labs(x = "Learning Rate",
       y = "Accuracy")
  # scale_x_log10()
learning_rate_plot2


learning_rate_plot3 <- ggplot(data =dat2 %>%
                                     filter(LR >= 0.25))+
  geom_point(aes(x = LR, y =TE_ACC))+
  theme_bw()+
  labs(x = "Learning Rate",
       y = "Accuracy")
learning_rate_plot3


momentum_plot3 <- ggplot(data =dat2)+
  geom_point(aes(x = MOMENTUM, y =TE_ACC, col = LR))+
  scale_color_viridis_c(trans = "log",
                        breaks = 10^seq(-4, 1, by = 1))+
  theme_bw()+
  labs(x = "Momentum",
       y = "Accuracy",
       col = "Learning\nRate")
momentum_plot3


pca <- prcomp(as.matrix(dat2), center = T)
plot(pca)

pca$rotation

corrplot2 <- function(data,
                      method = "pearson",
                      sig.level = 0.05,
                      order = "original",
                      diag = FALSE,
                      type = "upper",
                      tl.srt = 90,
                      number.font = 1,
                      number.cex = 1,
                      mar = c(0, 0, 0, 0)) {
  library(corrplot)
  data_incomplete <- data
  data <- data[complete.cases(data), ]
  mat <- cor(data, method = method)
  cor.mtest <- function(mat, method) {
    mat <- as.matrix(mat)
    n <- ncol(mat)
    p.mat <- matrix(NA, n, n)
    diag(p.mat) <- 0
    for (i in 1:(n - 1)) {
      for (j in (i + 1):n) {
        tmp <- cor.test(mat[, i], mat[, j], method = method)
        p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
      }
    }
    colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
    p.mat
  }
  p.mat <- cor.mtest(data, method = method)
  col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
  corrplot(mat,
           method = "color", col = col(200), number.font = number.font,
           mar = mar, number.cex = number.cex,
           type = type, order = order,
           addCoef.col = "black", # add correlation coefficient
           tl.col = "black", tl.srt = tl.srt, # rotation of text labels
           # combine with significance level
           p.mat = p.mat, sig.level = sig.level, insig = "blank",
           # hide correlation coefficiens on the diagonal
           diag = diag
  )
}

corrplot2(
  data = dat,
  method = "pearson",
  sig.level = 0.05,
  order = "original",
  diag = F,
  type = "upper",
  tl.srt = 90
)



