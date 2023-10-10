#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:oart2.py
# author:施惠_Collapsor
# datetime:2023/10/10 16:45
# software: PyCharm

"""
this is function  description 
"""

# import module your need

import os
import tqdm as tqdm
from PIL import Image
import numpy as np
import time
import pandas as pd
import tqdm

def read_img(path):
    try:
        img = Image.open(path)
        img = img.crop(img_bbox[path])
        img = img.resize((100,100))
        img.show()
        i_array = np.array(img)
        return i_array
    except Exception:
        print("read_img Error!")

def read_bbox():
    try:
        bbox_file = open(bbox_path, 'r')
    except Exception:
        print("list_bbox.txt IOError!")
    bbox_data = bbox_file.readlines()
    for d, i in zip(bbox_data[2:],range(bbox_data.__len__()-2)):
        s = d.split()
        p = data_path + s[0]
        bbox_loca = []
        for a in s[1:]:
            bbox_loca.append(float(a))
        img_bbox[p] = bbox_loca
    bbox_file.close()
    print('list_bbox.txt ready!')

def get_attr_type():
    attr_type_file = open(attr_type_path, 'r')
    lines = attr_type_file.readlines()
    for i in lines[2:]:
        s = i.split()
        attr = s[:-1]
        attr = ' '.join(attr)
        attr_num = s[-1]
        attr_type[attr] = attr_num
    print('list_attr_cloth.txt ready!')

def add_delete(dataframe, to_add, to_delete=[]):
    dataframe[to_add] = 0
    for x in to_delete:
        try:
            dataframe[to_add] += dataframe[x]
            # del dataframe[x]
        except:
            print("Error with: ", x)
    dataframe[to_add] = dataframe[to_add].replace(range(1, len(to_delete) + 1), 1)

def extract_column(dataframe, word):
    ls = []
    for x in dataframe.columns:
        if word in x:
            ls.append(x)
    print(ls)
    return ls

def order_col(dataframe, show_list = 30):
    print(dataframe.sum().sort_values(ascending=False)[:show_list])

def clean_df(dataframe):
    for col in dataframe:
        if "!" not in col:
            del dataframe[col]

def total_count(dataframe, show_count = True, delete=False):
    dataframe['total'] = dataframe.sum(axis=1,numeric_only = True)
    if show_count == True:
        print(dataframe.groupby(by = 'total').count())
    if delete == True:
        print("Before length :", len(dataframe))
        return dataframe.drop(dataframe[dataframe.total == 0].index)
        print("After length :", len(dataframe))

def drop_lower(dataframe, count):
    del dataframe['total']
    dataframe['total'] = dataframe.sum(axis=1)
    return dataframe.drop(dataframe[dataframe.total < count].index)


if __name__ == '__main__':
    data_path = 'Img/'
    bbox_path = 'list_bbox.txt'
    attr_path = 'list_attr_img.txt'
    # attr_path2 = directory + 'Anno_fine/list_attr_img.txt'
    attr_type_path = 'list_attr_cloth.txt'
    img_category_path = 'list_category_img.txt'
    # img_category_path2 = directory + 'Anno/list_category_img2.txt'

    img_category = {}
    img_attr = {}
    img_bbox = {}
    attr_type = {}
    attr_frequency = [0 for i in range(1000)]
    train_data = []
    test_data = []
    val_data = []

    get_attr_type()
    print(attr_type)

    df_attr = pd.DataFrame()
    df_attr = df_attr.from_dict(attr_type.items())

    df_attr.columns = ['Attribute', 'Type']
    print(df_attr)

    df_attr.groupby(by='Type').count()
    df_attr = df_attr.sort_values(['Attribute'], ascending=True)
    print(df_attr.columns)

    df_columns = ['pic']
    for i in df_attr['Attribute']:
        df_columns.append(i)
    print(df_columns)

    df = pd.read_csv(attr_path, delim_whitespace=True, header=None, names=df_columns)
    df.head()

    df = df.drop(labels=range(2))
    print(df)

    df.set_index('pic', inplace=True)
    print(df)

    df = df.astype(int)
    print(df)

    df2 = pd.read_csv(img_category_path, delim_whitespace=True, header=None, names=['category'])
    print(df2)

    df2 = df2.drop(labels='image_name')
    df2['upper_lower'] = ['1' if int(i) < 21 else '3' if int(i) > 36 else '2' for i in df2['category']]
    df2.groupby('upper_lower').count()
    print(df2)

    df['item_type'] = df2['category']
    df['upper_lower'] = df2['upper_lower']
    dff = df.sample(frac=0.1, random_state=123, axis=0)
    print(dff)

    dff = dff.replace(-1, 0)
    print(dff)

    df2f = pd.DataFrame(dff, columns=['item-type', 'upper_lower'])
    df2f.rename(columns={'item-type': 'category'}, inplace=True)
    # df2f=pd.DataFrame(columns = dff.columns)
    #
    # df2f = df2f.append(dff['item-type'])
    # df2f = df2f.append(dff['upper_lower'])
    # df2f['category']=dff['item-type']
    # df2f['upper_lower']=dff['upper_lower']
    # df2f.set_index('pic', inplace=True)
    print(df2f)

    df_upper = dff[dff['upper_lower'] == '1']
    df_lower = dff[dff['upper_lower'] == '2']

    d_texture = []
    d_fabric = []
    d_shape = []
    d_part = []
    d_style = []

    count = 0
    for i, r in df_attr.iterrows():
        x = r["Type"]
        y = r['Attribute']
        if x == "1":
            d_texture.append(y)
        elif x == "2":
            d_fabric.append(y)
        elif x == "3":
            d_shape.append(y)
        elif x == "4":
            d_part.append(y)
        else:
            d_style.append(y)

    df_texture_u = df_upper[d_texture]
    df_fabric_u = df_upper[d_fabric]
    df_shape_u = df_upper[d_shape]
    df_part_u = df_upper[d_part]
    df_style_u = df_upper[d_style]

    df_texture_l = df_lower[d_texture]
    df_fabric_l = df_lower[d_fabric]
    df_shape_l = df_lower[d_shape]
    df_part_l = df_lower[d_part]
    df_style_l = df_lower[d_style]

    df_upper2 = df_upper.copy()
    df_lower2 = df_lower.copy()
    df_texture = dff[d_texture]
    df_fabric = dff[d_fabric]
    df_shape = dff[d_shape]
    df_part = dff[d_part]
    df_style = dff[d_style]

    ##
    df_texture_u2 = df_texture_u.copy()
    df_fabric_u2 = df_fabric_u.copy()
    df_shape_u2 = df_shape_u.copy()
    df_part_u2 = df_part_u.copy()
    df_style_u2 = df_style_u.copy()

    ##
    df_texture_l2 = df_texture_l.copy()
    df_fabric_l2 = df_fabric_l.copy()
    df_shape_l2 = df_shape_l.copy()
    df_part_l2 = df_part_l.copy()
    df_style_l2 = df_style_l.copy()

    df_texture_u = df_upper[d_texture]
    df_texture_l = df_lower[d_texture]
    df_texture_u2 = df_texture_u.copy()
    df_texture_l2 = df_texture_l.copy()

    order_col(df_texture_u)
    print("")
    order_col(df_texture_l)

    # paisley = extract_column(df_texture, 'paisley')
    # add_delete(df_texture_u2, 't_paisley!', paisley)
    # add_delete(df_texture_l2, 't_paisley!', paisley)
    #
    floral = extract_column(df_texture, 'floral')
    add_delete(df_texture_u2, 't_floral!', floral)
    add_delete(df_texture_l2, 't_floral!', floral)
    #
    stripe = extract_column(df_texture, 'stripe')
    add_delete(df_texture_u2, 't_stripe!', stripe)
    add_delete(df_texture_l2, 't_stripe!', stripe)
    #
    dot = extract_column(df_texture, 'dot')
    add_delete(df_texture_u2, 't_dot!', dot)
    add_delete(df_texture_l2, 't_dot!', dot)
    #
    # tribal = extract_column(df_texture, 'tribal')
    # add_delete(df_texture_u2, 't_tribal!', tribal)
    # add_delete(df_texture_l2, 't_tribal!', tribal)
    #
    # zigzag = extract_column(df_texture, 'zig')
    # add_delete(df_texture_u2, 't_zigzag!', zigzag)
    # add_delete(df_texture_l2, 't_zigzag!', zigzag)
    #
    # add_delete(df_texture_u2, 't_print!', ['print','printed', 'graphic'])
    # add_delete(df_texture_l2, 't_print!', ['print','printed', 'graphic'])
    #
    # abstract = extract_column(df_texture, 'abstract')
    # add_delete(df_texture_u2, 't_abstract!', abstract)
    # add_delete(df_texture_l2, 't_abstract!', abstract)
    #
    # add_delete(df_texture_u2, 't_animal!', ['leopard','animal'])
    # add_delete(df_texture_l2, 't_animal!', ['leopard','animal'])

    clean_df(df_texture_u2)
    clean_df(df_texture_l2)

    order_col(df_texture_u2)

    total_count(df_texture_u2)
    print(len(df_texture_u2))

    order_col(df_texture_l2)

    total_count(df_texture_l2)
    print(len(df_texture_l2))

    # for the df_fabric(upper)

    order_col(df_fabric_u2)

    lace = extract_column(df_fabric, 'lace')
    add_delete(df_fabric_u2, 'f_lace!', lace)
    #
    # knit = extract_column(df_fabric,'knit')
    # add_delete(df_fabric_u2, 'f_knit!', knit)
    #
    denim = extract_column(df_fabric, 'denim')
    add_delete(df_fabric_u2, 'f_denim!', denim)
    #
    chiffon = extract_column(df_fabric, 'chiffon')
    add_delete(df_fabric_u2, 'f_chiffon!', chiffon)
    #
    cotton = extract_column(df_fabric, 'cotton')
    add_delete(df_fabric_u2, 'f_cotton!', cotton)
    #
    leather = extract_column(df_fabric, 'leather')
    add_delete(df_fabric_u2, 'f_leather!', leather)
    #
    # pleated = extract_column(df_fabric, 'pleat')
    # add_delete(df_fabric_u2, 'f_pleated!', pleated )
    #
    fur = extract_column(df_fabric, 'fur')
    add_delete(df_fabric_u2, 'f_fur!', leather)
    #
    # sheer = extract_column(df_fabric, 'sheer')
    # add_delete(df_fabric_u2, 'f_sheer!', sheer)
    #
    # embroidered = extract_column(df_fabric, 'embroidered')
    # add_delete(df_fabric_u2, 'f_embroidered!', embroidered)

    clean_df(df_fabric_u2)

    order_col(df_fabric_u2)

    total_count(df_fabric_u2)
    print(len(df_fabric_u2))

    df_fabric_l2 = df_fabric_l.copy()
    order_col(df_fabric_l2)

    # For the d_fabric(lower)
    df_fabric_l2 = df_fabric_l.copy()
    order_col(df_fabric_l2)

    denim = extract_column(df_fabric_l, 'denim')
    add_delete(df_fabric_l2, 'f_denim!', denim)
    #
    # wash = extract_column(df_fabric_l, 'wash')
    # add_delete(df_fabric_l2, 'f_wash!', wash)
    #
    # add_delete(df_fabric_l2, 'f_distressed/ripped!', ['distressed','ripped'])
    #
    leather = extract_column(df_fabric_l, 'leather')
    add_delete(df_fabric_l2, 'f_leather!', leather)
    #
    cotton = extract_column(df_fabric_l, 'cotton')
    add_delete(df_fabric_l2, 'f_cotton!', cotton)
    #
    knit = extract_column(df_fabric_l, 'knit')
    add_delete(df_fabric_l2, 'f_knit!', knit)
    #
    pleated = extract_column(df_fabric_l, 'pleat')
    add_delete(df_fabric_l2, 'f_pleated!', pleated)

    clean_df(df_fabric_l2)

    order_col(df_fabric_l2)
    total_count(df_fabric_l2)
    print(len(df_fabric_l2))

    # For df_shape(upper)

    df_shape_u2 = df_shape_u.copy()
    order_col(df_shape_u, 20)
    clean_df(df_shape_u2)
    order_col(df_shape_u2)
    total_count(df_shape_u2)
    print(len(df_shape_u2))

    # For ds_shape (lower)
    df_shape_l2 = df_shape_l.copy()
    order_col(df_shape_l, 20)

    fit = extract_column(df_shape_l, 'fit')
    add_delete(df_shape_l2, 's_fit!', fit + ['slim'])
    #
    pencil = extract_column(df_shape_l, 'pencil')
    add_delete(df_shape_l2, 's_pencil!', pencil)
    #
    # capri = extract_column(df_shape_l, 'capri')
    # add_delete(df_shape_l2, 's_capri!', capri)
    #
    midi = extract_column(df_shape_l, 'midi')
    add_delete(df_shape_l2, 's_midi!', midi)
    #
    mini = extract_column(df_shape_l, 'mini')
    add_delete(df_shape_l2, 's_mini!', mini)
    #
    maxi = extract_column(df_shape_l, 'maxi')
    add_delete(df_shape_l2, 's_maxi!', maxi)

    clean_df(df_shape_l2)
    order_col(df_shape_l2)

    total_count(df_shape_l2)
    print(len(df_shape_l2))

    df_part_u2 = df_part_u.copy()
    order_col(df_part_u, 50)

    sleeve = extract_column(df_part_u, 'sleeve')
    sleeveless = extract_column(df_part_u, 'sleeveless')
    sleeve = [x for x in sleeve if x not in sleeveless]
    # add_delete(df_part_u2, 'p_sleeve!', sleeve)
    add_delete(df_part_u2, 'p_sleeveless!', sleeveless)
    #
    l_sleeve = extract_column(df_part_u, 'long')
    add_delete(df_part_u2, 'p_long-sleeve!', l_sleeve)
    #
    collar = extract_column(df_part_u, 'collar')
    collarless = extract_column(df_part_u, 'collarless')
    collar = [x for x in collar if x not in collarless]
    add_delete(df_part_u2, 'p_collar!', collar)
    # add_delete(df_part_u2, 'p_collarless!', collarless)
    #
    pocket = extract_column(df_part_u, 'pocket')
    add_delete(df_part_u2, 'p_pocket!', pocket)
    #
    vneck = extract_column(df_part_u, 'v-neck')
    add_delete(df_part_u2, 'p_v-neck!', vneck)
    #
    button = extract_column(df_part_u, 'button')
    add_delete(df_part_u2, 'p_button!', button)
    #
    hood = extract_column(df_part_u, 'hood')
    add_delete(df_part_u2, 'p_hooded!', hood)
    #
    zipper = extract_column(df_part_u, 'zip')
    add_delete(df_part_u2, 'p_zipper!', zipper)

    clean_df(df_part_u2)
    order_col(df_part_u2)
    total_count(df_part_u2)
    print(len(df_part_u2))

    df_part_l2 = df_part_l.copy()
    order_col(df_part_l)

    zipper = extract_column(df_part_u, 'zip')
    add_delete(df_part_l2, 'p_zipper!', zipper)

    clean_df(df_part_l2)
    order_col(df_part_l2)
    total_count(df_part_l2)
    print(len(df_part_l2))

    df_style = df[d_style]
    df_style2 = df_style.copy()

    order_col(df_style_u)

    order_col(df_style_l)

    summer = extract_column(df_style, 'summer')
    add_delete(df_style_u2, 'y_summer!', summer)
    add_delete(df_style_l2, 'y_summer!', summer)
    #
    basic = extract_column(df_style, 'basic')
    add_delete(df_style_u2, 'y_basic!', basic)
    add_delete(df_style_l2, 'y_basic!', basic)
    #
    cute = extract_column(df_style, 'cute')
    add_delete(df_style_u2, 'y_cute!', cute)
    add_delete(df_style_l2, 'y_cute!', cute)
    #
    chic = extract_column(df_style, 'chic')
    add_delete(df_style_u2, 'y_chic!', chic)
    add_delete(df_style_l2, 'y_chic!', chic)
    #
    retro = extract_column(df_style, 'retro')
    add_delete(df_style_u2, 'y_retro!', retro)
    add_delete(df_style_l2, 'y_retro!', retro)
    #
    dark = extract_column(df_style, 'dark')
    add_delete(df_style_u2, 'y_dark!', dark)
    add_delete(df_style_l2, 'y_dark!', dark)
    #
    # yoga = extract_column(df_style, 'yoga')
    # add_delete(df_style_u2, 'y_yoga!', yoga)
    # add_delete(df_style_l2, 'y_yoga!', yoga)
    #
    athletic = ['workout', 'running', 'run', 'athletic']
    add_delete(df_style_u2, 'y_athletic!', athletic)
    add_delete(df_style_l2, 'y_athletic!', athletic)

    clean_df(df_style_u2)
    order_col(df_style_u2)
    total_count(df_style_u2)
    print(len(df_style_u2))

    clean_df(df_style_l2)
    order_col(df_style_l2)
    total_count(df_style_l2)
    print(len(df_style_l2))

    print("df_upper:")
    df_upper.head()
    print("df_lower:")
    df_lower.head()

    del df_upper['item_type']
    del df_upper['upper_lower']
    del df_lower['item_type']
    del df_lower['upper_lower']

    df_old_u = df_upper.copy()
    df_old_l = df_lower.copy()
    df_upper = df_old_u.copy()
    df_lower = df_old_l.copy()

    # df2_string = df2.to_string()
    ls_upper = [df_texture_u2, df_fabric_u2, df_part_u2]  # shape and style ommited
    # , df_upper['item_type!'], df_upper['upper_lower!']
    # for x in ls_upper:
    #    del x['total']
    df_upper = pd.concat(ls_upper, axis=1)
    df_upper.head()
    df_upper2 = df_upper.copy()

    ls_lower = [df_texture_l2, df_fabric_l2, df_shape_l2, df_part_l2]
    # , df_upper['item_type!'], df_upper['upper_lower!']
    # for x in ls_upper:
    #    del x['total']
    df_lower = pd.concat(ls_lower, axis=1)
    df_lower.head()
    df_lower2 = df_lower.copy()

    print("Number of Attributes")
    print("upper: ", len(df_upper.columns))
    print("lower: ", len(df_lower.columns))

    print("Number of Entries")
    print("upper: ", len(df_upper))
    print("lower: ", len(df_lower))

    print(df_upper.columns)
    print(df_lower.columns)

    del df_upper['total']
    total_count(df_upper)

    del df_lower['total']
    total_count(df_lower)

    df_upper.rename(columns=lambda x: x.split("!")[0], inplace=True)
    df_lower.rename(columns=lambda x: x.split("!")[0], inplace=True)

    # df_upper = pd.merge(df_upper, df2, left_index = True, right_index = True)
    # df_lower = pd.merge(df_lower, df2, left_index = True, right_index = True)

    df_upper.to_csv('category-upper-all-everything.txt', sep=':')
    df_lower.to_csv('category-lower-all-everything.txt', sep=':')

    df_upper = total_count(df_upper, show_count=False, delete=True)
    df_lower = total_count(df_lower, show_count=False, delete=True)

    print("After length : ", len(df_upper))
    print("After length : ", len(df_lower))

    df_upper.head()

    df_upper2 = df_upper.copy()
    df_lower2 = df_lower.copy()

    df_upper_al2 = drop_lower(df_upper2, 2)
    del df_upper_al2['total']
    df_upper_al3 = drop_lower(df_upper2, 3)
    del df_upper_al3['total']

    print(len(df_upper2))
    print(len(df_upper_al2))
    print(len(df_upper_al3))
    print(len(df_upper2))
    order_col(df_upper2)
    print(len(df_upper_al2))
    order_col(df_upper_al2)
    print(len(df_upper_al3))
    order_col(df_upper_al3)

    df_lower_al2 = drop_lower(df_lower2, 2)
    del df_lower_al2['total']
    df_lower_al3 = drop_lower(df_lower2, 3)
    del df_lower_al3['total']

    print(len(df_lower2))
    print(len(df_lower_al2))
    print(len(df_lower_al3))

    print(len(df_lower2))
    order_col(df_lower2)

    print(len(df_lower_al2))
    order_col(df_lower_al2)

    print(len(df_lower_al3))
    order_col(df_lower_al3)

    df2f.head()

    df_upper_al2_final = pd.merge(df_upper_al2, df2, left_index = True, right_index = True)
    df_lower_al2_final = pd.merge(df_lower_al2, df2, left_index = True, right_index = True)

    df_upper_al3.to_csv('category-upper-all.txt', sep=':')
    df_lower_al2.to_csv('category-lower-all.txt', sep=':')

    print(len(df_upper_al2_final.columns))
    print(len(df_lower_al2_final.columns))
    print(df_upper_al2_final.columns)
    print(df_lower_al2_final.columns)

    print("end")

