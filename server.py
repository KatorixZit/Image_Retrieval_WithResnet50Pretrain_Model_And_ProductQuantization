import numpy as np
from feature_extractor import FeatureExtractor
from datetime import datetime
import os
from tensorflow.keras.preprocessing import image as kimage
from PIL import Image

from flask_sqlalchemy import SQLAlchemy
from flask import Flask, url_for, render_template, request, redirect, session

from scipy.cluster.vq import vq, kmeans2
# from flask_ngrok import run_with_ngrok

# Read image features
fe = FeatureExtractor()


# def cosine_similarity(query, X):
#     norm_2_query = np.sqrt(np.sum(query*query))
#     norm_2_X = np.sqrt(np.sum(X*X, axis=-1))
#     return np.sum(query*X, axis=-1)/(norm_2_query*norm_2_X)

# def retrieval_images(query_vector, imgs_feature):
#     # caculate similarity between query and features in database
#     rates = cosine_similarity(query_vector, imgs_feature)
#     id_s = np.argsort(-rates)[:100] # Top 30 results
    
#     return [(round(rates[id], 2), paths_feature[id]) for id in id_s]


# def compute_code_books(vectors, sub_size=2, n_cluster=128, n_iter=20, minit='points', seed=123):
#     n_rows, n_cols = vectors.shape
#     n_sub_cols = n_cols // sub_size

#     np.random.seed(seed)
#     code_books = np.zeros((sub_size, n_cluster, n_sub_cols), dtype=np.float32)
#     for subspace in range(sub_size):
#         sub_vectors = vectors[:, subspace * n_sub_cols:(subspace + 1) * n_sub_cols]
#         centroid, label = kmeans2(sub_vectors, n_cluster, n_iter, minit=minit)
#         code_books[subspace] = centroid

#     return code_books


def encode(vectors, code_books):
    n_rows, n_cols = vectors.shape
    sub_size = code_books.shape[0]
    n_sub_cols = n_cols // sub_size

    codes = np.zeros((n_rows, sub_size), dtype=np.int32)
    for subspace in range(sub_size):
        sub_vectors = vectors[:, subspace * n_sub_cols:(subspace + 1) * n_sub_cols]
        code, dist = vq(sub_vectors, code_books[subspace])
        codes[:, subspace] = code

    return codes


def query_dist_table(query, code_books):
    sub_size, n_cluster, n_sub_cols = code_books.shape

    dist_table = np.zeros((sub_size, n_cluster))
    for subspace in range(sub_size):
        sub_query = query[subspace * n_sub_cols:(subspace + 1) * n_sub_cols]

        diff = code_books[subspace] - sub_query.reshape(1, -1)
        diff = np.sum(diff ** 2, axis=1)
        dist_table[subspace, :] = diff

    return dist_table


def retrieval_images(query_vector, imgs_feature):
    M=16
    pqcode = encode(imgs_feature, codebooks) 
    dist_table = query_dist_table(query_vector, codebooks)

    # lookup the distance
    dists = np.sum(dist_table[range(M), pqcode], axis=1)
    # the numpy indexing trick is equivalent to the following loop approach
    n_rows = pqcode.shape[0]
    dists = np.zeros(n_rows).astype(np.float32)
    for n in range(n_rows):
        for m in range(M):
            dists[n] += dist_table[m][pqcode[n][m]]
    nearest = np.argsort(dists)[:30] # Top 30 results
    
    return [(round(dists[id],2), paths_feature[id]) for id in nearest]



folder_query = "query_pic/"
root_fearure_path = "static/feature/all_feartures.npz"
codebooks_path = "static/feature/codebooks.npz"

data = np.load(root_fearure_path)
paths_feature = data["array1"]
imgs_feature = data["array2"]

data_codebook = np.load(codebooks_path)
codebooks = data_codebook["array1"]

app = Flask(__name__)
# run_with_ngrok(app)
app.secret_key = "super secret key"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    """ Create user table"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password = db.Column(db.String(80))

    def __init__(self, username, password):
        self.username = username
        self.password = password

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login Form"""
    if request.method == 'GET':
        return render_template('login.html', mess ="")
    else:
        name = request.form['username']
        passw = request.form['password']

        data = User.query.filter_by(username=name, password=passw).first()
        if data is not None:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            # tài khoản sai
            return render_template('login.html', mess= "nhap sai account")


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register Form"""
    if request.method == 'POST':
        new_user = User(
            username=request.form['username'],
            password=request.form['password'])

        users = User.query.all()
        for user in users:
            if user.username == new_user.username:
                # username da ton tai
                return render_template('register.html', mess= "username da ton tai")

        db.session.add(new_user)
        db.session.commit()
        return render_template('login.html')
    else:
        return render_template('register.html')


@app.route('/', methods=['GET', 'POST'])
def index():

    try:
        if session['logged_in'] == False:
            return redirect(url_for('login'))
    except Exception:
        return redirect(url_for('login'))


    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        
        # Load query image and FeatureExtractor
        query = kimage.load_img(uploaded_img_path, target_size=(224, 224))
        query = kimage.img_to_array(query, dtype=np.float32)
        query_vector = fe.extract(query[None, :]).flatten()

        # retrieval_images
        scores = retrieval_images(query_vector, imgs_feature)


        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores1=scores[:10],
                               scores2=scores[10:20],
                               scores3=scores[20:])

    return render_template('index.html')


if __name__=="__main__":
    # app.run(host='192.168.1.34', port='6868', debug=False)
    app.run(host='localhost', port='6868', debug=False)
    # app.run()
