# this script classifies each region as DSB or not

DSB <- read.table("DSB_hotspots_ID.txt")
encode <- read.table("encode_roadmap_act_col1.txt")

newcol <- "DSB"
DSB[,newcol] <- 1

encode[,newcol] <- 0
head(encode)

df <- rbind(DSB, encode)
summary(df)
table(df$DSB)

write.table(df, "activity.txt", sep="\t", quote = FALSE, row.names = FALSE)
