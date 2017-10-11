

def gcd(n1,n2):
    
    #if n1==0: return [0,0,n2]
    #if n2==0: return [0,n1,0]
    #if n1==1 or n1==1.0: return [n2,1,n2]
    #if n2==1 or n2==1.0: return [n1,n1,1]
    x=1
    while (abs(n1)<1):
        n1*=10
        n2*=10
        x*=10
    num1 = n1
    num2 = n2
    while num2 != 0:
        numRem = num1 % num2
        num1 = num2
        num2 = numRem
    num1 = abs(num1)/x
    n1 = n1/num1/x
    n2 = n2/num1/x
    return [num1,n1,n2]
    #return [int(num1),int(n1),int(n2)]