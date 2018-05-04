from tsf.windows import SimpleAR

time_serie = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

ar = SimpleAR(n_prev=5)
X = ar.transform(X=[], y=time_serie)
y = ar.offset_y(X, time_serie)

print X
print y
