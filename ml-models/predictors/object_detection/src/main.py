from predict_treecounting import predict_main
from params import MODEL_PATH, INPUT_PATH, OUTPUT_PATH

if __name__ == '__main__':
    predict_main(INPUT_PATH, MODEL_PATH, OUTPUT_PATH, bound_path=None, out_type="bbox",verbose=1)
