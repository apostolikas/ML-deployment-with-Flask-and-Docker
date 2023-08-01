import tensorflow as tf
from model import SimpleNet
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = SimpleNet.build()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

# Evaluate the model on the test data using `evaluate`
print("\nEvaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=128)
print(f"Test loss : {results[0]}")
print(f"Test accuracy : {results[1]}")

save_path = './model.h5'
model.save(save_path)
