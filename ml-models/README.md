# Run machine learning model in docker container

## Prerequisites

- conda or miniconda
- docker and nvidia-docker2 install on your system. Please read https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- login with private registry:

```
docker login http://registry.eofactory.ai:5000 -u admin -p 123654123
```

## Coding your model

- Make a new folder `type-of-model/model-name` in root project folder
- Scaffold your model

  ```
  cp -r example-model type-of-model/model-name
  ```

- Prepare your local environment
  - Install base environment using conda
    ```
    conda env create --name model-name --file=base-images/base-image-name/environment.yaml && cd type-of-model/model-name && conda activate model-name
    ```
  - Install additional dependencies
  - Update Dockerfile
- Declare your input in params.py and .env file
- Prepare a folder where contains your model data like .h5 file, etc on your local machine. This folder path must be named like `/local-root-folder/type-of-model/model-name/data-version`
- Start coding your model

## Deploy your model

- Copy local data to network drive with naming rule: `/network-driver-root-folder/type-of-model/model-name/data-version`
- Push your code to git
- Tag your version

  ```
  git tag type-of-model/model-name@semantic-version
  ```

  Example:

  ```
  git tag pretrain-models/green-cover@1.0.0
  ```

  You can tag multi models in a commit

- Inside folder `/network-driver-root-folder/type-of-model/model-name`, create `{semantic-version}-testing.txt` file that contain environment variables for testing
- Push your tag and CI-CD server will automatically build your model
  ```
  git push --tags
  ```
