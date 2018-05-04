from tsf.windows import DinamicWindow

time_series =   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],                    # Endogenous serie
                 [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],          # Exogenous serie #1
                 [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]          # Exogenous serie #2

dw = DinamicWindow(stat='variance', ratio=0.1, metrics=['variance', 'mean'], indexs=[0, 2])
X = dw.transform(X=[], y=time_series)
y = dw.offset_y(X, time_series)

print X
print y