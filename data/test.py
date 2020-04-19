import numpy as np
import pandas as pd

#test = open("DSB_hotspots_ID.txt")

df = open("test.txt")
row_list = []
for line in df:
	line = line.strip()
	my_list = [line]
	row_list.append(my_list)


#for index, rows in df.iterrows():
	#my_list = [rows.hi]
	#row_list.append(my_list)

#print(row_list)


#df = pd.concat([df, pd.DataFrame(columns = list(col_add))])
#print()





m = np.identity(len(row_list), dtype = int)
print(m)

trying = pd.DataFrame(data=m, index = range(len(df)), columns = list())
print(trying)


