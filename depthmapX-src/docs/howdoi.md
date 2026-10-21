Calculate NACh and NAIn
---
NACh (Normalised Angular Choice) and NAIn (Normalised Angular Integration) were introduced in Hillier et al. (2012).
To create them first calculate Choice and Total depth (Tools -> Segment -> Run Angular Segment Analysis..., include choice), create the two new attributes/columns (Attributes -> Add Column) and finally set their formulas (Attribute -> Update Column).

According to Hillier et al. (2012) the equation for NACh is:  
**`NACh = log( ACH + 1) / log( ATD + 3)`**  
where `ACH = Angular Choice` and `ATD = Angular Total Depth`.  
Thus the formula in depthmapX needs to be:  
**`log(value("T1024 Choice") + 1) / log(value("T1024 Total Depth") + 3)`**,  
assuming that the analysis carried out was Angular Tulip analysis with 1024 bins and no radius limit.

The equivalent for NAIn is:  
**`NAIn = NC ^ 1.2 / ATD`**  
where `NC = Node Count` and `ATD = Angular Total Depth`.  
Thus the formula for depthmapX needs to be:  
**`value("T1024 Node Count")^1.2 / value("T1024 Total Depth")`**  


References:
- Hillier, W. R. G., Yang, T., & Turner, A. (2012). Normalising least angle choice in Depthmap - And how it opens up new perspectives on the global and local analysis of city space. Journal of Space Syntax, 3(2), 155–193.
