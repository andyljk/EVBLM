library(EVBLM)
set.seed(248)
cases=list(small=matrix(rnorm(100*600),100), tall=matrix(rnorm(160*128),160),
           wide=matrix(rnorm(128*160),128), constant=matrix(1,128,160))
Q=qr.Q(qr(matrix(rnorm(128*3),128))); V=qr.Q(qr(matrix(rnorm(160*3),160)))
cases$tied=Q%*%diag(c(5,5,1))%*%t(V)
cases$close=Q%*%diag(c(5,5-1e-5,1))%*%t(V)
for(name in names(cases)) {
 X=cases[[name]]
 columns=split(seq_len(ncol(X)),ceiling(seq_len(ncol(X))/4))
 data=lapply(columns,function(j)X[,j,drop=FALSE]); D=lapply(columns,seq_along)
 before=.Random.seed
 z=evblm(data,D,R=1,fn='EC',max_iter=0,null_check=FALSE)
 stopifnot(identical(.Random.seed,before))
 u=z$u$pos[,1,1]; v=unlist(lapply(z$v$pos,function(x)x[1,,1]))
 sigma=sqrt(sum(u^2)*sum(v^2)); u=u/sqrt(sum(u^2));v=v/sqrt(sum(v^2))
 exact=svd(X,nu=1,nv=1)
 residual=max(sqrt(sum((X%*%v-sigma*u)^2)),sqrt(sum((t(X)%*%u-sigma*v)^2)))/sigma
 error=abs(sigma/exact$d[1]-1)
 angle=1-abs(sum(u*exact$u[,1]))
 stopifnot(error<1e-9,residual<1e-8)
 if(name!='tied')stopifnot(abs(angle)<1e-8)
}
