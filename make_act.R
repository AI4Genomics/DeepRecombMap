# this script classifies each region as DSB or not

DSB <- read.table("DSB_hotspots_ID.txt")
encode <- read.table("encode_roadmap_act_col1.txt")

newcol <- "DSB"
DSB[,newcol] <- 1

reg <- "regulatory"
DSB[,reg] <- 0

encode[,newcol] <- 0
encode[,reg] <- 1

df <- rbind(DSB, encode)
summary(df)
table(df$DSB)
table(df$reg)
write.table(df, "activity.txt", sep="\t", quote = FALSE, row.names = FALSE)
