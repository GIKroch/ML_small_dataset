# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:54:41 2019

@author: xx
"""

# NOTATKI

y = data1.SalePrice
X = data1.drop("SalePrice", axis=1)
X = sm.add_constant(X)  # Adds a constant term to the predictor

est = sm.OLS(y, X)
est = est.fit()
est.summary()

y_star = est.predict()

fig,ax = plt.subplots()
ax.scatter(data1.SalePrice, y_star)
ax.plot([y_star.min(), y_star.max()], [y_star.min(), y_star.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
fig.show()

y = data2.logSalePrice
X = data2[list(final.keys())]
X = sm.add_constant(X)  # Adds a constant term to the predictor

est = sm.OLS(y, X)
est = est.fit()
est.summary()


y_star = est.predict()

fig,ax = plt.subplots()
ax.scatter(data2.logSalePrice, y_star)
ax.plot([y_star.min(), y_star.max()], [y_star.min(), y_star.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
fig.show()


y_star = np.exp(est.predict())

fig,ax = plt.subplots()
ax.scatter(np.exp(data2.logSalePrice), y_star)
ax.plot([y_star.min(), y_star.max()], [y_star.min(), y_star.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
fig.show()
