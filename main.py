from src.test.fold_eval import fold_evaluate

#reg_list = ['pH','Available.P']
#reg_list = ['CEC', 'Exchangeable.Ca', 'Exchangeable.K','Exchangeable.Mg']
#reg_list = ['CEC', 'Humus']
#reg_list = ['EC', 'NH4.N','NO3.N','Inorganic.N']


reg_list = [['pH','crop'],
            ['CEC','crop'],
            ['EC','crop'],
            ['NH4.N','crop'],
            ['NO3.N','crop'],
            ['Inorganic.N','crop'],
            ['Humus','crop']]

exclude_ids = [
    '041_20_Sait_Carr', '042_20_Sait_Eggp', '043_20_Sait_Carr', '044_20_Sait_Broc',
    '045_20_Sait_Broc', '046_20_Sait_Burd', '047_20_Sait_Burd', '048_20_Sait_Yama',
    '049_20_Sait_Yama', '050_20_Sait_Stra', '061_20_Naga_Barl', '062_20_Naga_Barl',
    '067_20_Naga_Pump', '331_22_Niig_jpea', '332_22_Niig_jpea'
]

#exclude_ids = ['a']

url1 = 'data\\raw\\taxon_data\\taxon_lv6.csv'
url2 = 'data\\raw\\chem_data.xlsx'

epoch = 200

for reg in reg_list:
    fold_evaluate(feature_path = url1, target_path = url2, reg_list = reg, exclude_ids = exclude_ids,
                    epochs = epoch, patience = 5, lr=0.0001,
                    output_dir = f'result_{epoch}_pat')

