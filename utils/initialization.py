def initialization(training_data, testing_data, n, m, d):
    training_data_length = training_data.index.size
    mu = training_data_length / n / m

    r = {}
    r_item = {}
    I_u = {}
    I = []
    for index, row in training_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        
        if not item_id in I:
            I.append(item_id)

        r.setdefault(usr_id)
        r_item.setdefault(item_id)
        I_u.setdefault(usr_id)

        if r[usr_id] == None:
            r[usr_id] = []
        if r_item[item_id] == None:
            r_item[item_id] = []
        if I_u[usr_id] == None:
            I_u[usr_id] = []
        
        if not item_id in r[usr_id]:
            r[usr_id].append(item_id)
        if not usr_id in r_item[item_id]:
            r_item[item_id].append(usr_id)
        if not item_id in I_u[usr_id]:
            I_u[usr_id].append(item_id)

    I_u_com = {}
    for u in range(1, n + 1):
        I_u_com.setdefault(u)
        if I_u_com[u] == None:
            I_u_com[u] = []
        if not u in I_u:
            I_u_com[u] = range(1, m + 1)
            continue
        for i in range(1, m + 1):
            if not i in I_u[u]:
                I_u_com[u].append(i)
    
    b_usr = {}
    for usr_id in range(1, n + 1):
        if not usr_id in r:
            b_usr[usr_id] = -mu
            continue
        b_usr[usr_id] = len(r[usr_id]) / m - mu

    b_item = {}
    for item_id in range(1, m + 1):
        if not item_id in r_item:
            b_item[item_id] = -mu
            continue
        b_item[item_id] = len(r_item[item_id]) / n - mu

    r_te = {}
    I_te = {}
    U_te = []
    for index, row in testing_data.iterrows():
        usr_id = row[0]
        item_id = row[1]

        if not usr_id in U_te:
            U_te.append(usr_id)

        r_te.setdefault(usr_id)
        I_te.setdefault(usr_id)

        if r_te[usr_id] == None:
            r_te[usr_id] = []
        if I_te[usr_id] == None:
            I_te[usr_id] = []

        if not item_id in r_te[usr_id]:
            r_te[usr_id].append(item_id)
        if not item_id in I_te[usr_id]:
            I_te[usr_id].append(item_id)
    
    U = (np.random.random((n,d)) - 0.5) * 0.01
    V = (np.random.random((m,d)) - 0.5) * 0.01
    W = (np.random.random((m,d)) - 0.5) * 0.01

    return r, I_u, I_u_com, I, r_te, I_te, U_te, b_usr, b_item, U, V, W
