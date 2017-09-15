import sys
url = sys.argv[1]

#read file ----
f = open(url,'r')
#print ('Name of the file: ', f.name)
strr = f.read()
#print (strr)  
	
#split text by space----
arrs =  strr.split()
#print (arr) 

#create list & initial----
elements = []
count = -1
add_ele = True

#loop----
for arr in arrs:
	for element in elements:
		if element[0]==arr:
			element[2] += 1
			add_ele = False
			break	
	if add_ele==True:
		elements.append([])
		count += 1
		elements[count].append(arr)
		elements[count].append(count)
		elements[count].append(1)
	add_ele = True
#print(elements)

#output file----
out = open('Q1.txt','w')
for ele in elements[:-1]:
	out.write(ele[0]+' ')
	rr = str(ele[1])
	out.write(rr+' ')
	rr = str(ele[2])
	out.write(rr+'\n')
#last no \n
out.write(elements[-1][0]+' ')
rr = str(elements[-1][1])
out.write(rr+' ')
rr = str(elements[-1][2])
out.write(rr)

#close file----
f.close()
out.close() 