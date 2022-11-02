---
> cora 1%
> 
> GCN 62.7+2.7  
> CLNode(GCN) 69.6+1.2 lam=0.75 T=100
> 
> GAT 65.2+2.4  
> CLNode 68.5+2.0
> 
> GraphSage 54.8+3.0    
> CLNode 61.8+2.6
> 
> JKNET 67.5+1.7    
> CLNode 69.4+1.4
> 

> cora 2%
> 
> GCN 73.5+0.8  
> CLNode(GCN) 77.0+0.7
> 
> GAT 74.2+1.2  
> CLNode 77.1+1.1
> 
> GraphSage  68.2+2.4   
> CLNode 72.1+1.4
> 
> JKNET   74.0+1.5  
> CLNode 76.8+0.8

> cora 4%
> 
> GCN 80.5+0.5    
> CLNode(GCN)   81.1+0.8
> 
> GAT 80.8+0.8    
> CLNode 80.1+0.8
> 
> GraphSage  78.4+0.7   
> CLNode+1.3
> 
> JKNET   79.1+1.1  
> CLNode+0.6



> CiteSeer
> 
> GCN 53.2+1.9  
> CLNode 58.2+1.7
> 
> GAT 55.2+2.2  
> CLNode 58.9+1.4
> 
> GraphSage 51.0+2.4    
> CLNode 54.4+2.0
> 
> JKNET 53.2+2.8    
> CLNode 57.0+2.2
> 

> pubmed
> 
> GCN 64.3+2.9  
> CLNode 65.9+1.3
> 
> GAT 64.6+2.5  
> CLNode 68.2+2.6
> 
> GraphSage 61.3+1.4    
> CLNode 64.1+3.8
> 
> JKNET 66.0+1.7    
> CLNode 62.3+3.5

> amazon-computers
> 
> GCN 79.0+3.7  
> CLNode 84.7+0.5
> 
> GAT 80.2+0.8  
> CLNode 82.6+1.1
> 
> GraphSage 71.7+2.4    
> CLNode 77.5+1.6
> 
> JKNET 83.2+1.3    
> CLNode 84.4+1.0


> amazon-photo
> 
> GCN 89.1+0.8  
> CLNode 90.8+1.0
> 
> GAT 89.4+1.8    
> CLNode 90.1+1.1
> 
> GraphSage  83.0+2.6   
> CLNode 87.5+1.2
> 
> JKNET  89.2+0.7   
> CLNode 90.4+0.9
> 

---
# noise
> ## cora
> 
> ### uniform
> 5%    
> GCN  78.1
> GAT  79.3
> CLNode(GCN) 80.6
> CLNode(GAT)   80.3
> 
> 10%     
> GCN  76.9
> GAT  77.8
> CLNode(GCN) 78.9
> CLNode(GAT)  79.5
> 
> 15%      
> GCN  76.8
> GAT 77.8
> CLNode(GCN) 77.8
> CLNode(GAT)  78.2
> 
> 20%      
> GCN 74.7
> GAT 77.0
> CLNode(GCN) 78.4
> CLNode(GAT)  77.5
> 
> 25%      
> GCN 70.2
> GAT 75.1
> CLNode(GCN) 75.0
> CLNode(GAT)  76.0
> 
> 30%   
> GCN 69.6
> GAT 71.2
> CLNode(GCN) 74.4
> CLNode(GAT)  74.7
> 
> ### pair
> GCN 78.5
> GAT 78.5
> CLNode(GCN) 79.5
> CLNode(GAT)  79.8
> 
> 10%      
> GCN 76.4
> GAT 77.8
> CLNode(GCN) 78.9
> CLNode(GAT)  78.6
> 
> 15%   
> GCN 72.9
> GAT 75.5
> CLNode(GCN) 77.6
> CLNode(GAT)  76.9
> 
> 20%   
> GCN 70.1
> GAT 73.2
> CLNode(GCN) 73.9
> CLNode(GAT)  74.9
> 
> 25%   
> GCN 69.0
> GAT 71.1
> CLNode(GCN) 72.7
> CLNode(GAT)  74.2
> 
> 30%   
> GCN 65.1
> GAT 66.0
> CLNode(GCN) 68.1
> CLNode(GAT)  69.9



> ## citeseer
> 
> ### uniform
> 5%    
> GCN 66.3
> GAT 66.5
> CLNode(GCN) 69.5
> CLNode(GAT)   69.4
> 
> 10%     
> GCN 66.0
> GAT 67.6
> CLNode(GCN) 68.3
> CLNode(GAT)   69.4
> 
> 15%      
> GCN 63.3
> GAT 65.8
> CLNode(GCN) 66.6
> CLNode(GAT)  68.5
> 
> 20%      
> GCN 62.3
> GAT 62.3
> CLNode(GCN) 65.7
> CLNode(GAT)  66.5
> 
> 25%      
> GCN 57.5
> GAT 61.6
> CLNode(GCN) 63.6
> CLNode(GAT)  63.7
> 
> 30%   
> GCN 55.1
> GAT 57.8
> CLNode(GCN) 63.1
> CLNode(GAT)  63.6
> 
> ### pair
> GCN 65.0
> GAT 68.1
> CLNode(GCN) 69.6
> CLNode(GAT)   69.6
> 
> 10%      
> GCN 62.8
> GAT 62.5
> CLNode(GCN) 67.8
> CLNode(GAT)  68.5
> 
> 15%   
> GCN 60.7
> GAT 61.9
> CLNode(GCN) 65.4
> CLNode(GAT)  66.5
> 
> 20%   
> GCN 60.2
> GAT 59.5
> CLNode(GCN) 64.8
> CLNode(GAT)  64.2
> 
> 25%   
> GCN 52.2
> GAT 57.0
> CLNode(GCN) 60.3
> CLNode(GAT)  63.3
> 
> 30%   
> GCN 47.9
> GAT 49.7
> CLNode(GCN) 58.4
> CLNode(GAT)  59.9


> ## pubmed
> 
> ### uniform
> 5%    
> GCN 76.5
> GAT 77.2
> CLNode(GCN) 78.9
> CLNode(GAT)  78.9
> 
> 10%     
> GCN 76.8
> GAT 77.0
> CLNode(GCN) 77.9
> CLNode(GAT)  77.5
> 
> 15%      
> GCN 74.9
> GAT 76.1
> CLNode(GCN) 77.3
> CLNode(GAT)  77.0
> 
> 20%      
> GCN 74.0
> GAT 74.1
> CLNode(GCN) 77.4
> CLNode(GAT)  77.1
> 
> 25%      
> GCN 74.8
> GAT 71.9
> CLNode(GCN) 76.1
> CLNode(GAT)  76.2
> 
> 30%   
> GCN 69.4
> GAT 69.3
> CLNode(GCN) 74.4
> CLNode(GAT)  75.2
> 
> ### pair
> GCN 76.6
> GAT 76.9
> CLNode(GCN) 78.5
> CLNode(GAT)  78.1
> 
> 10%      
> GCN 74.1
> GAT 75.5
> CLNode(GCN) 76.8
> CLNode(GAT)  77.5
> 
> 15%   
> GCN 70.7
> GAT 74.8
> CLNode(GCN) 75.3
> CLNode(GAT)  76.0
> 
> 20%   
> GCN 68.7
> GAT 69.1
> CLNode(GCN) 73.9
> CLNode(GAT)  75.2
> 
> 25%   
> GCN 65.6
> GAT 69.5
> CLNode(GCN) 71.3
> CLNode(GAT)  72.8
> 
> 30%   
> GCN 67.7
> GAT 62.9
> CLNode(GCN) 69.0
> CLNode(GAT)  72.0
> 


> ## a-computers
> 
> ### uniform
> 5%    
> GCN 81.5
> GAT 83.0
> CLNode(GCN) 82.3
> CLNode(GAT)  83.9
> 
> 10%     
> GCN 80.2
> GAT 82.8
> CLNode(GCN) 82.1
> CLNode(GAT)  83.8
> 
> 15%      
> GCN 79.4
> GAT 82.2
> CLNode(GCN) 81.0
> CLNode(GAT)  82.9
> 
> 20%      
> GCN 78.6
> GAT 81.0
> CLNode(GCN) 80.2
> CLNode(GAT)  81.3
> 
> 25%      
> GCN 74.9
> GAT 80.2
> CLNode(GCN) 79.4
> CLNode(GAT)  80.9
> 
> 30%   
> GCN 68.3
> GAT 79.5
> CLNode(GCN) 76.3
> CLNode(GAT)  80.8
> 
> ### pair
> GCN 81.4
> GAT 83.2
> CLNode(GCN) 82.4
> CLNode(GAT)  83.8
> 
> 10%      
> GCN 73.0
> GAT 81.9
> CLNode(GCN) 81.2
> CLNode(GAT)  82.9
> 
> 15%   
> GCN 70.3
> GAT 79.5
> CLNode(GCN) 81.3
> CLNode(GAT)  81.5
> 
> 20%   
> GCN 70.0
> GAT 77.7
> CLNode(GCN) 79.8
> CLNode(GAT)  81.6
> 
> 25%   
> GCN 68.2
> GAT 72.9
> CLNode(GCN) 73.2
> CLNode(GAT)  79.0
> 
> 30%   
> GCN 50.6
> GAT 72.2
> CLNode(GCN) 73.8
> CLNode(GAT)  76.5

> ## a-photo
> 
> ### uniform
> 5%    
> GCN  91.6
> GAT 90.2
> CLNode(GCN) 91.9
> CLNode(GAT)  91.4
> 
> 10%     
> GCN 88.6
> GAT 88.9
> CLNode(GCN) 90.0
> CLNode(GAT)  91.4
> 
> 15%      
> GCN 88.0
> GAT 88.6
> CLNode(GCN) 90.7
> CLNode(GAT)  89.7
> 
> 20%      
> GCN 87.9
> GAT 88.3
> CLNode(GCN) 88.6
> CLNode(GAT)  90.4
> 
> 25%      
> GCN 88.3
> GAT 87.1
> CLNode(GCN) 87.8
> CLNode(GAT)  88.7
> 
> 30%   
> GCN 84.5
> GAT 85.2
> CLNode(GCN) 86.6
> CLNode(GAT)  88.1
> 
> ### pair
> GCN 90.9
> GAT 90.7
> CLNode(GCN) 91.3
> CLNode(GAT)  90.7
> 
> 10%      
> GCN 89.5
> GAT 88.3
> CLNode(GCN) 90.1
> CLNode(GAT)  90.0
> 
> 15%   
> GCN 86.6
> GAT 88.0
> CLNode(GCN) 89.3
> CLNode(GAT)  88.8
> 
> 20%   
> GCN 83.4
> GAT 85.3
> CLNode(GCN) 87.9
> CLNode(GAT)  87.1
> 
> 25%   
> GCN 83.0
> GAT 84.0
> CLNode(GCN) 84.4
> CLNode(GAT)  87.4
> 
> 30%   
> GCN 78.0
> GAT 78.7
> CLNode(GCN) 83.5
> CLNode(GAT)  83.0