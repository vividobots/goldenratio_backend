from m2r import rft_arr as n,rft1_arr as r


a=[]
Name=['Distance between the eyes',
'Width of right eye ',
'Width of left eye ',
'End of the arc to length of the right eyebrow',
'End of the arc to length of the left eyebrow',
'Right mouth edge to side of the face ',
'Left mouth edge to side of the face',
'Center of mouth to chin',
'Width of upper lip',
'Width of the nose',
'Width of forehead',
'Width of the chin']

Name2=['Distance from right eye inner edge to side of face',
'Distance from left eye inner edge to side of face',
'Width of the right eyebrow',
'Width of the left eyebrow',
'Length of the nose',
'Width of the mouth',
'Starting of the nose to center of the mouth',
'Width of lower lip',
'Right eye inner edge to cheekbone',
'Left eye inner edge to cheekbone',
'Centre of forehead to right side of face',
'Centre of forehead to left side of face' ]
def cal ():
 for i, j,r1,r2 in zip(n,r,Name,Name2):
    p=i*1.618
    q=p/j
    if q*100>100:
        q=j/p
        q*100
        print(f"{r1},{r2}\n{i,j},percentage= {q*100}\n")
        a.append(q*100)
    else:
        q*100
        print(f"{r1},{r2}\n{i,j},percentage= {q*100}\n")
        a.append(q*100)

 avg = sum(a)/ len(a)
#print(f"\033[33m{avg}\033[0m")
 return avg

org=cal()
print(org)
print(len(a))




