import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import cv2 as cv

digits = sklearn.datasets.load_digits()

m = len(digits.images)
data = digits.images.reshape((m, -1))

x_train, x_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
)

x_train = x_train / 16.0
x_test = x_test / 16.0

encoder = OneHotEncoder(sparse_output=False, categories='auto')
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

input_layer_size = 64  
hidden_layer_size = 16
output_layer_size = 10  

theta1 = np.random.random((input_layer_size + 1, hidden_layer_size)) * 2 / input_layer_size
theta2 = np.random.random((hidden_layer_size + 1, output_layer_size)) * 2 / hidden_layer_size

def sigmoid(n):
    return 1 / (1 + np.exp(-n))

def sigmoid_gradient(n):
    return n * (1 - n)

def relu(n):
    return np.maximum(0, n)

def relu_gradient(n):
    return np.where(n > 0, 1, 0)

def cost_function(pred, res, m, lambda_):
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    regularization = (lambda_ / (2 * m)) * (
        np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2)
    )
    J = -(1 / m) * np.sum(res * np.log(pred) + (1 - res) * np.log(1 - pred)) + regularization
    return J

def forward_pass(x, theta1, theta2):

    a1 = np.c_[np.ones(x.shape[0]), x]

    z2 = np.matmul(a1, theta1)
    a2 = relu(z2)
    a2 = np.c_[np.ones(a2.shape[0]), a2]  
    
    z3 = np.matmul(a2, theta2)
    a3 = relu(z3)
    
    return a3, a2, a1

def back_propagation(x, y, theta1, theta2, alpha):
    m = x.shape[0] 
    
    output, a2, a1 = forward_pass(x, theta1, theta2)
    
    output_layer_error = (output - y) * relu_gradient(output)
    hidden_layer_error = np.matmul(output_layer_error, theta2[1:, :].T) * relu_gradient(a2[:, 1:])
    
    theta2_gradient = np.matmul(a2.T, output_layer_error) / m
    theta1_gradient = np.matmul(a1.T, hidden_layer_error) / m
    
    theta2 -= alpha * theta2_gradient
    theta1 -= alpha * theta1_gradient
    
    return theta1, theta2

for epoch in range(8500):
    theta1, theta2 = back_propagation(x_train, y_train, theta1, theta2, 0.085)

def predict(x, theta1, theta2):
    pred, _, _ = forward_pass(x, theta1, theta2)
    return np.argmax(pred, axis=1)

predictions = predict(x_test, theta1, theta2)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == true_labels) * 100

print('Accuracy: ', accuracy)

img = np.full((64, 64, 3), 0, dtype=np.uint8)
resized = np.full((8, 8, 3), 0, dtype=np.uint8)

def draw(event, x, y, flags, param):
    global mouse_down, resized, theta1, theta2
    if event == cv.EVENT_LBUTTONDOWN:
        mouse_down = True
    if event == cv.EVENT_LBUTTONUP:
        mouse_down = False
    if event == cv.EVENT_MOUSEMOVE and mouse_down:
        cv.circle(img, (x, y), 3, (200, 200, 200), -1)
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray_image, (8, 8), interpolation= cv.INTER_LINEAR)
        cv.imshow('Original', gray_image)
        print(predict(convert_to_brightness(resized), theta1, theta2), end='\r')

def convert_to_brightness(rgb_img):
    rgb_img = np.reshape(rgb_img, (1, 64))
    return rgb_img / 140

cv.namedWindow('Original')
cv.setMouseCallback('Original', draw)
mouse_down = False

cv.imshow('Original', img)

while True:
    cv.imshow('Original', img)
    k = cv.waitKey(0)
    if k == ord('q'):
        cv.destroyAllWindows()
        break
    if k == ord('r'):
        img = np.full((64, 64, 3), 0, dtype=np.uint8)
        resized = np.full((8, 8, 3), 0, dtype=np.uint8)

