from tsf.windows import SimpleAR, DinamicWindow
from tsf.pipeline import TSFPipeline

time_series = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],            # Endogenous serie
               [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]  # Exogenous serie #1

pipe = TSFPipeline([('ar', SimpleAR(n_prev=2)),
                    ('dw', DinamicWindow(stat='variance', ratio=0.1, metrics=['variance', 'mean'])),
                    ('', None)])

X, y = pipe.transform(X=[], y=time_series)

print X
print y