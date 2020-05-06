import xlrd
import numpy as np
import matplotlib.pyplot as plt

workbook = xlrd.open_workbook('jester-data-3.xls')
worksheet = workbook.sheet_by_name('jester-data-3-new')

#read data

#we want to predict the value of cell[:,5] (because this column is densely filled) using all other 99 columns
ROWS = 24938
CLOUMNS = 100
X = np.zeros(shape = (ROWS, 100))
Y = np.zeros(shape = ROWS)
for i in range(ROWS):
    for j in range(100):
        if j != 5:
            if worksheet.cell(i, j).value != 99:
                X[i,j] = worksheet.cell(i, j).value
        else:
            if worksheet.cell(i, j).value != 99:
                Y[i] = worksheet.cell(i, j).value

            



def gradient_ascent(f, betas, init_step, iterations):
    val, grad = f(betas) #use function linear_regression_log_likelihood to compute gradient and y value
    vals = [val]
    for i in range (iterations):
        done = False
        count = 0
        step = init_step
        while not done and count < iterations:
            new_betas = betas + step * grad
            new_val, new_gard = f(new_betas)
            if new_val < val: #overshot
                step = step * 0.95
                count += 1
            else:
                done = True
        
        if not done:
            print("gradient ascent failed.")
        else:
            val = new_val
            betas = new_betas
            vals.append(val)
    return val, betas
    
def linear_regression_log_likelihood(X, Y, betas, sigma2 = 1.0):
    ll = 0
    beta0 = betas[0] 
    betas = betas[1:]
    deltabeta0 = 0
    deltabeta = np.zeros(len(betas))    
    for (x,y) in zip(X,Y):
        ll = ll -0.5 * np.log(2 * np.pi * sigma2)        
        res = y - beta0 - np.sum(x * betas)
        ll = ll - 1.0 / (2.0 * sigma2) * (res ** 2.0)
        deltabeta0 = deltabeta0 - 1.0 / sigma2 * res * (-1)
        deltabeta = deltabeta - 1.0 / sigma2 * (res * (-x))
    grad = np.zeros(len(betas) + 1)
    grad[0] = deltabeta0
    grad[1:] = deltabeta
    return ll, grad
          
def calculate_accurancy(Y, X, betas):
    differences = []
    realAndPredicted = []
    for i in range(ROWS):
        if Y[i] != 0: # means this row's fifth column is valid
            predicted = betas[0] + np.sum(X[i] * betas[1:])
            difference = Y[i] - predicted
            differences.append(difference)
            realAndPredicted.append([Y[i], predicted])
    ply.plot(realAndPredicted)
    plt.xlebel("Actual values")
    plt.ylabel("Predicted values")
    return np.mean(differences)
    

init_beta = [0.1]*101 #includes one space for beta0
init_step = 0.01
iterations = 2000
f = lambda betas : linear_regression_log_likelihood(X, Y, betas)
[f_best, betas_mle] = gradient_ascent(f, init_beta, init_step, iterations)

average_difference = calculate_accurancy(Y, X, betas_mle)
print("Averange difference = " + str(average_difference))

for i in range(100):
    print("Beta " + str(i) + " = " + str(betas_mle[i]))
    
