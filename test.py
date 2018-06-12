from evaluation import evaluate

def train_oxford_test_paris(des_type, output_file, sv=True, qe=True):

    mAP = evaluate("./paris/6k", "./oxford/5k", des_type,
                   "./paris/gt", "oxford", sv_enable=sv, qe_enable=qe)
    ans = "{} trained on Oxford got {} mAP on Paris".format(des_type, mAP)

    with open(output_file, 'a') as f:
        print(ans, file=f)

    return ans

def train_paris_test_oxford(des_type, output_file, sv=True, qe=True):

    mAP = evaluate("./oxford/5k", "./paris/6k", des_type,
                   "./oxford/gt", "paris", sv_enable=sv, qe_enable=qe)
    ans = "{} trained on Paris got {} mAP on Oxford".format(des_type, mAP)

    with open(output_file, 'a') as f:
        print(ans, file=f)

    return ans

if __name__ == "__main__":

    descriptors = ['l2net',
                   'HardNetAll',
                   'surf',
                   # 'DeepCompare',
                   'sift']

    d = {'BoW': (False, False),
         'SV': (True, False)#,
         }#'SV+QE': (True, True)}

    for mode_name, mode_params in d.items():

        OUTPUT_FILE = 'results_{}.txt'.format(mode_name)

        with open(OUTPUT_FILE, 'a') as f:
            print("####################", file=f)
            for des_type in descriptors:
                train_oxford_test_paris(des_type, OUTPUT_FILE, *mode_params)
                train_paris_test_oxford(des_type, OUTPUT_FILE, *mode_params)