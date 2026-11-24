# ETIA find_adjset steps (run_adjset_dagitty_r.R) for many graphs; pcalg2dagitty copied from pcalg 2.7-12 (GPL-2+).
# Rscript adjset_dagitty.R in.json out.json   (R_LIBS=~/R/library: dagitty 0.3.4, jsonlite)
suppressMessages(library(dagitty)); library(jsonlite)
pcalg2dagitty <- function(amat,labels,type="cpdag")
{
  if (type == "cpdag" | type=="pdag") {
      result <- "pdag{"
  }  else if (type=="dag") {
      result <- "dag{"  
  } else if (type=="mag") {
      result <- "mag{"
  } else if (type=="pag") {
      result <- "pag{"
  } 
  
  n <- length(amat[1,]) # =? ncol(amat) == #{nodes}
  
  ## if type of graph is DAG/CPDAG then edges are denoted with 0 and 1
  if (type %in% c("pdag","dag","cpdag")){
    
    for (i in 1:(n-1))
    {
      ## for single (unconnected) nodes
      if (sum(amat[,i]+amat[i,])==0){
        result <- paste0(result," ",labels[i]," ")
      }
      for(j in (i+1) : n)
      {
        if (amat[i,j] & amat[j,i]) {
          result <- paste(result,labels[i],"--",labels[j],sep=" ")
        } else if (amat[i,j] & !amat[j,i]) {
          result <- paste(result,labels[i],"<-",labels[j],sep=" ")
        } else if (!amat[i,j] & amat[j,i]) {
          result <- paste(result,labels[i],"->",labels[j],sep=" ")
        }
      }
    }
    
  } else if (type %in% c("mag","pag")) {
    
    ## if the type of graph is MAG/PAG then edgemarks are encoded with
    ## 1,2 and 3 in the adjacency matrix
      
      for (i in 1:(n-1))
      {
        ##for single nodes
        if (sum(amat[,i]+amat[i,])==0){
          result <- paste0(result," ",labels[i]," ")
        }
        for(j in (i+1) : n)
        {
          if (amat[i,j]==1 & amat[j,i]==1){
            result <- paste(result,labels[i],"@-@",labels[j],sep=" ")
          } else if (amat[i,j]==2 & amat[j,i]==3) {
            result <- paste(result,labels[i],"->",labels[j],sep=" ")
          } else if (amat[i,j]==3 & amat[j,i]==2) {
            result <- paste(result,labels[i],"<-",labels[j],sep=" ")
          } else if (amat[i,j]==3 & amat[j,i]==3) {
            result <- paste(result,labels[i],"--",labels[j],sep=" ")
          } else if (amat[i,j]==2 & amat[j,i]==2) {
            result <- paste(result,labels[i],"<->",labels[j],sep=" ")
          } else if (amat[i,j]==2 & amat[j,i]==1) {
            result <- paste(result,labels[i],"@->",labels[j],sep=" ")
          } else if (amat[i,j]==1 & amat[j,i]==2) {
            result <- paste(result,labels[i],"<-@",labels[j],sep=" ")
          }
        }
      }
  }
  ## if last node is unconnected
  if (sum(amat[,n]+amat[n,])==0){
    result <- paste0(result," ",labels[n]," ")
  }
  result <- paste0(result,"}")
  
  ## transform the resulting string into a daggity object
  dagitty::dagitty(result)
}
args <- commandArgs(trailingOnly=TRUE)
cases <- fromJSON(args[1], simplifyVector=FALSE)
first <- function(r){ r <- unclass(r); if (length(r) == 0) return(NULL); s <- r[[1]]; if (length(s) == 0) character(0) else s }
out <- lapply(cases, function(cs){
  amat <- matrix(unlist(cs$amat), nrow=length(cs$labels), byrow=TRUE)
  t0 <- Sys.time()
  res <- tryCatch({
    g <- pcalg2dagitty(amat, labels=unlist(cs$labels), type=cs$type)
    can <- adjustmentSets(g, exposure=unlist(cs$exposures), outcome=unlist(cs$outcomes), type="canonical", effect="total")
    mini <- adjustmentSets(g, exposure=unlist(cs$exposures), outcome=unlist(cs$outcomes), type="minimal", effect="total")
    list(canonical=first(can), minimal=first(mini), n_minimal=length(unclass(mini)))
  }, error=function(e) list(error=conditionMessage(e)))
  res$time <- as.numeric(Sys.time() - t0, units="secs")
  res
})
writeLines(toJSON(out, auto_unbox=TRUE, null="null"), args[2])
