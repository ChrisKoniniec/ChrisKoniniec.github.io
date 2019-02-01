---
layout: post
title: Customer Segmentation with KMeans
nocomments: true
categories: [python, unsupervised, machine learning, clustering, PCA]
---

Customer Segmentation
====================
Hello again and welcome to another one of my projects! This time, the boss has given us a dataset with previous transaction data and is asking for a few different things:

- Separate data into 3 different tables (Customers/Orders/Products) and write to a SQL database for storage.
- Create a Recency-Frequency-Monetary table for reporting.
- Segment the customers into groups using a basic KMeans clustering algorithm.

These clusters could represent anything from pushing different advertisements to those customers to inviting them back with a deal if they haven't bought from our site in a while, or even general demographic information for later use. We will be evaluating our model using silhouette score, Calinski-Harabaz score, and distortion score. Lets get to it!

# Step 1: Primary Analysis and Cleaning

After importing our libraries, we'll read in the data from the single csv file that we've been given.


```python
#import data:
df = pd.read_csv("customers.csv")

df.info()
```
<details>
  <summary>List of features </summary>

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 24958 entries, 0 to 24957
    Data columns (total 74 columns):
    Customers.id                     24950 non-null object
    Customers.fname                  8572 non-null object
    Customers.lname                  8572 non-null object
    Customers.create_date            8572 non-null object
    Customers.mailing                8572 non-null object
    Customers.last_modified          8572 non-null object
    Orders.id                        8572 non-null object
    Orders.fname                     8572 non-null object
    Orders.lname                     8572 non-null object
    Orders.order_number              8572 non-null object
    Orders.currency                  8572 non-null object
    Orders.subtotal                  8572 non-null float64
    Orders.shipping                  8572 non-null float64
    Orders.total                     8572 non-null object
    Orders.shipping_carrier          8571 non-null object
    Orders.shipping_method           8564 non-null object
    Orders.tracking                  8564 non-null object
    Orders.payment_status            8564 non-null float64
    Orders.payment_date              8564 non-null float64
    Orders.payment_type              8564 non-null object
    Orders.payment_amount            8564 non-null float64
    Orders.payment_id                8564 non-null object
    Orders.payment_code              8564 non-null object
    Orders.status                    8564 non-null float64
    Orders.placed_date               8564 non-null float64
    Orders.updated_date              8564 non-null float64
    Orders.shipped_date              8564 non-null float64
    Order_Items.id                   8564 non-null float64
    Order_Items.product_id           8564 non-null float64
    Order_Items.product_name         8564 non-null object
    Order_Items.qty                  8564 non-null float64
    Order_Items.price                8564 non-null float64
    Order_Items.cost                 8564 non-null float64
    Products.id                      8564 non-null float64
    Products.template                8564 non-null object
    Products.vendor                  8564 non-null float64
    Products.import_id               8564 non-null float64
    Products.name                    8564 non-null object
    Products.display_name            8564 non-null object
    Products.list_price              8564 non-null float64
    Products.price                   8564 non-null float64
    Products.cost                    8564 non-null float64
    Products.flags                   8564 non-null float64
    Products.last_modified           8564 non-null float64
    Products.taxable                 8564 non-null float64
    Products.shopping_gtin           8564 non-null float64
    Products.shopping_brand          8564 non-null object
    Products.shopping_mpn            8564 non-null object
    Products.shopping_flags          8564 non-null float64
    Products.amazon_asin             8564 non-null object
    Products.amazon_item_type        8564 non-null object
    Products.google_shopping_id      8564 non-null object
    Products.google_shopping_type    8564 non-null object
    Products.google_shopping_cat     8564 non-null object
    Products.shopping_type           8564 non-null object
    Products.pricegrabber_cat        8564 non-null object
    Products.thefind_cat             8564 non-null object
    Products.quickbooks_id           8564 non-null object
    Products.qb_edit_sequence        8564 non-null float64
    Products.short_description       8564 non-null object
    Products.long_description        8557 non-null object
    Products.seo_title               4186 non-null object
    Products.seo_url                 4186 non-null object
    Products.unit                    4186 non-null object
    Products.packaging               4186 non-null object
    Products.multiple                4186 non-null object
    Products.upc                     4186 non-null float64
    Products.hcpcs                   4186 non-null object
    Products.case_qty                4186 non-null float64
    Products.import_flags            4186 non-null float64
    Products.shipping_length         4186 non-null float64
    Products.shipping_width          4186 non-null float64
    Products.shipping_height         4186 non-null float64
    Products.family_id               4186 non-null object
    dtypes: float64(32), object(42)
    memory usage: 14.1+ MB
</details>

As we can see, there are quite a few different kinds of data in this table. Since our first task is to separate our data into categories and write them to a SQL server, we'll see if there's an easy ID number we can use to split everything off.

```python
#Since we're going to separate the tables, check to see if we can do so easily
#by separating on ID, no such luck
df[['Customers.id', 'Orders.id', 'Products.id']].head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customers.id</th>
      <th>Orders.id</th>
      <th>Products.id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>797</td>
      <td>3758</td>
      <td>2310.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>23</td>
      <td>177.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9531</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>29</td>
      <td>983.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>30</td>
      <td>991.0</td>
    </tr>
  </tbody>
</table>
</div>

No such luck, let's keep exploring the data and see what else we can find out about the different tables that we'll be creating.

```python
pd.DataFrame([{'customers': len(df['Customers.id'].value_counts()),
               'products': len(df['Products.id'].value_counts()),    
               'orders': len(df['Orders.id'].value_counts()),  
              }], columns = ['products', 'orders', 'customers'], index = ['quantity'])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>products</th>
      <th>orders</th>
      <th>customers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>quantity</th>
      <td>1710</td>
      <td>3568</td>
      <td>3058</td>
    </tr>
  </tbody>
</table>
</div>

So there are 3054 unique customers, who made 3568 unique orders of 1710 unique products. That will be useful information for later on. While exploring, I found there are a bunch of nonsense rows. Lets take those out.

```python
df['Customers.id'].replace('\t<li>Commode liners can be used with most commode buckets.</li>', np.NaN, inplace = True)
df['Customers.id'].replace('\t<li>Liners include an absorbent pad which solidifies the waste and makes clean up easy and hygienic.</li>',
                                  np.NaN, inplace = True)


df.dropna(thresh= 8, inplace = True)
#we have about 70 columns, so if any observations are missing 10% or more of the data we should drop them

df.reset_index(inplace=True, drop=True)
```

As a final step in exploration, lets take a quick look at some of the transactions for each customer.

```python
#Lets take a quick look at our customers and the order prices
Customers_groupdf = df.groupby(['Customers.id', 'Customers.fname', 'Customers.lname'])['Order_Items.price',
                                                                                       'Order_Items.cost', 'Orders.total'].sum()

Customers_groupdf.head(20)
```


<details>
  <summary>Customers groupby table</summary>


  <div>
  <style scoped>
      .dataframe tbody tr th:only-of-type {
          vertical-align: middle;
      }

      .dataframe tbody tr th {
          vertical-align: top;
      }

      .dataframe thead th {
          text-align: right;
      }
  </style>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th></th>
        <th></th>
        <th>Order_Items.price</th>
        <th>Order_Items.cost</th>
      </tr>
      <tr>
        <th>Customers.id</th>
        <th>Customers.fname</th>
        <th>Customers.lname</th>
        <th></th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>3.0</th>
        <th>John</th>
        <th>Smith</th>
        <td>73.78</td>
        <td>54.37</td>
      </tr>
      <tr>
        <th>4.0</th>
        <th>James</th>
        <th>Anderson</th>
        <td>19.56</td>
        <td>12.62</td>
      </tr>
      <tr>
        <th>5.0</th>
        <th>Abraham</th>
        <th>Pollak</th>
        <td>95.14</td>
        <td>66.33</td>
      </tr>
      <tr>
        <th>7.0</th>
        <th>peggy</th>
        <th>thompson</th>
        <td>39.19</td>
        <td>27.99</td>
      </tr>
      <tr>
        <th>8.0</th>
        <th>Randy</th>
        <th>Pruss</th>
        <td>59.75</td>
        <td>45.96</td>
      </tr>
      <tr>
        <th>10.0</th>
        <th>Tommy</th>
        <th>Smith</th>
        <td>34.00</td>
        <td>34.00</td>
      </tr>
      <tr>
        <th>11.0</th>
        <th>Mark</th>
        <th>Tremble</th>
        <td>34.00</td>
        <td>34.00</td>
      </tr>
      <tr>
        <th>12.0</th>
        <th>Emely</th>
        <th>Cooke</th>
        <td>10.76</td>
        <td>2.82</td>
      </tr>
      <tr>
        <th>13.0</th>
        <th>george</th>
        <th>mcmillin</th>
        <td>118.68</td>
        <td>85.22</td>
      </tr>
      <tr>
        <th>14.0</th>
        <th>adrian</th>
        <th>Cavitt</th>
        <td>339.99</td>
        <td>339.99</td>
      </tr>
      <tr>
        <th>15.0</th>
        <th>Sharon</th>
        <th>Mueller</th>
        <td>18.94</td>
        <td>11.32</td>
      </tr>
      <tr>
        <th>21.0</th>
        <th>Corey</th>
        <th>Edmondson</th>
        <td>34.00</td>
        <td>34.00</td>
      </tr>
      <tr>
        <th>22.0</th>
        <th>Robert</th>
        <th>Miller</th>
        <td>6.84</td>
        <td>1.71</td>
      </tr>
      <tr>
        <th>23.0</th>
        <th>Mekala</th>
        <th>Whitaker</th>
        <td>141.40</td>
        <td>100.40</td>
      </tr>
      <tr>
        <th>24.0</th>
        <th>Richard</th>
        <th>Ariano</th>
        <td>29.38</td>
        <td>20.26</td>
      </tr>
      <tr>
        <th>25.0</th>
        <th>marc</th>
        <th>gorzynski</th>
        <td>35.00</td>
        <td>35.00</td>
      </tr>
      <tr>
        <th>26.0</th>
        <th>Richard L.</th>
        <th>Shaak</th>
        <td>35.00</td>
        <td>35.00</td>
      </tr>
      <tr>
        <th>27.0</th>
        <th>Kenneth</th>
        <th>Schmude</th>
        <td>125.16</td>
        <td>31.29</td>
      </tr>
      <tr>
        <th>30.0</th>
        <th>Jesse</th>
        <th>Spalding</th>
        <td>35.00</td>
        <td>35.00</td>
      </tr>
      <tr>
        <th>31.0</th>
        <th>Alan</th>
        <th>Safir</th>
        <td>56.78</td>
        <td>43.68</td>
      </tr>
    </tbody>
  </table>
  </div>
</details>

# Step 2: Separate into 3 DataFrames

In this step we'll separate each chunk of our original table, clean them up and at the end, write it all into a SQL server.

```python
#Customers DataFrame
customers = df.loc[: , :'Customers.last_modified']
customers.columns = [col.split('.')[1] for col in customers.columns]

#Orders DataFrame, we need to specify columns because they start with both:
#Orders.colname and Order_Items.colname which sometimes overlap
orders = df.loc[: , 'Orders.id':'Order_Items.cost']
orders_cols = ['id', 'fname', 'lname', 'order_number', 'currency', 'subtotal', 'shipping', 'total',  'shipping_carrier',
               'shipping_method', 'tracking', 'payment_status', 'payment_date', 'payment_type', 'payment_amount', 'payment_id',
               'payment_code', 'status', 'placed_date', 'updated_date', 'shipped_date', 'Items.id', 'Items.product_id',
               'Items.product_name', 'Items.qty', 'Items.price', 'Items.cost']
orders.columns = orders_cols

#Products DataFrame
products = df.loc[: , 'Products.id':]
products.columns = [col.split('.')[1] for col in products.columns]
```

## Customers Table

Separating out each factor from the larger table creates a bunch of duplicates, so our primary step for each table will be dropping duplicates with the same ID.

```python
#As you can see from this example, there are many duplicates
customers[customers['id'] == 3371.0]
```

<details>
  <summary>Customer duplicates</summary>

  <div>
  <style scoped>
      .dataframe tbody tr th:only-of-type {
          vertical-align: middle;
      }

      .dataframe tbody tr th {
          vertical-align: top;
      }

      .dataframe thead th {
          text-align: right;
      }
  </style>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>id</th>
        <th>fname</th>
        <th>lname</th>
        <th>create_date</th>
        <th>mailing</th>
        <th>last_modified</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>3821</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3822</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3823</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3824</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3825</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3826</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3827</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3828</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3829</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3830</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3831</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3832</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3833</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3834</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3835</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3836</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3837</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3838</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3839</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3840</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3841</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3842</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3843</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3844</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3845</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3846</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3847</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3848</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3849</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>3850</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>...</th>
        <td>...</td>
        <td>...</td>
        <td>...</td>
        <td>...</td>
        <td>...</td>
        <td>...</td>
      </tr>
      <tr>
        <th>8162</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8163</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8164</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8165</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8166</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8167</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8168</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8169</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8170</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8171</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8172</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8173</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8174</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8175</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8176</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8177</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8178</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8179</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8180</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8181</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8182</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8183</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8184</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8185</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8186</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8187</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8188</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8189</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8190</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
      <tr>
        <th>8191</th>
        <td>3371</td>
        <td>Terry</td>
        <td>Rich</td>
        <td>1461637787</td>
        <td>1.0</td>
        <td>1461637787</td>
      </tr>
    </tbody>
  </table>
  <p>4371 rows × 6 columns</p>
  </div>
</details>

```python
customers.drop_duplicates(subset='id', inplace = True)

#After doing this, there is one more ID that is null, we will drop this as well
customers.drop(index = [801], inplace = True)
```

Our next step will be converting the columns to correct data types so that we can work with that data in python later if we need. This doesn't matter so much for writing to the SQL database because we can specify formatting later on.

```python
#Converting some columns to correct data type
customers['create_date'] = customers['create_date'].astype(int)
customers['mailing'] = customers['mailing'].astype(float)
customers['last_modified'] = customers['last_modified'].astype(int)
customers['create_date'] = customers['create_date'].transform(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
customers['last_modified'] = customers['last_modified'].transform(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
```

Lets just take one more look at the info to make sure we're good to go.

```python
customers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3054 entries, 0 to 8571
    Data columns (total 6 columns):
    id               3054 non-null object
    fname            3054 non-null object
    lname            3054 non-null object
    create_date      3054 non-null object
    mailing          3054 non-null float64
    last_modified    3054 non-null object
    dtypes: float64(1), object(5)
    memory usage: 167.0+ KB

## Orders Table

As before, we will first drop the duplicates. Then we drop the remaining rows with null values, about 3 rows in this table.

```python
orders.drop_duplicates(subset='id', inplace = True)
orders.dropna(inplace = True)
```

Then we do the feature transformations similar to the previous table.
```python
orders['total'] = orders['total'].astype(float)
orders['updated_date'] = orders['updated_date'].transform(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
orders['shipped_date'] = orders['shipped_date'].transform(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
```

Next, we'll create a column 'profit' that will be useful for segmentation later. This is also useful to get a quick snapshot of how much our company is making from each transaction.

```python
orders['profit'] = (orders['Items.price'] - orders['Items.cost']) / orders['Items.qty']
orders['profit'].describe()
```

    count    3565.000000
    mean       15.067519
    std        22.067572
    min       -29.510000
    25%         6.010000
    50%        10.620000
    75%        18.000000
    max       549.000000
    Name: profit, dtype: float64

Already we the benefit of creating this column is apparent, there are some orders we are actually LOSING money on! We can sort by those transactions that our profit is less than 0 for further analysis later on, although we won't be covering it in this specific project, we have to stay focused on the task at hand!

```python
#we can take a look at which orders we're losing money on and why
orders[orders['profit'] < 0]
```


<details>
  <summary>Orders we lose money on</summary>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>fname</th>
      <th>lname</th>
      <th>order_number</th>
      <th>currency</th>
      <th>subtotal</th>
      <th>shipping</th>
      <th>total</th>
      <th>shipping_carrier</th>
      <th>shipping_method</th>
      <th>tracking</th>
      <th>payment_status</th>
      <th>payment_date</th>
      <th>payment_type</th>
      <th>payment_amount</th>
      <th>payment_id</th>
      <th>payment_code</th>
      <th>status</th>
      <th>placed_date</th>
      <th>updated_date</th>
      <th>shipped_date</th>
      <th>Items.id</th>
      <th>Items.product_id</th>
      <th>Items.product_name</th>
      <th>Items.qty</th>
      <th>Items.price</th>
      <th>Items.cost</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>812</th>
      <td>15142</td>
      <td>Marcia</td>
      <td>Olsen</td>
      <td>15142</td>
      <td>USD</td>
      <td>29.97</td>
      <td>0.00</td>
      <td>29.97</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.76035E+11</td>
      <td>3.0</td>
      <td>1.459800e+09</td>
      <td>authorize.net</td>
      <td>29.97</td>
      <td>8133230477</td>
      <td>03161Z</td>
      <td>1.0</td>
      <td>1.459800e+09</td>
      <td>2016-04-04 15:39:40</td>
      <td>2016-04-04 15:39:40</td>
      <td>17707.0</td>
      <td>1454.0</td>
      <td>Sterile Bordered Gauze</td>
      <td>3.0</td>
      <td>9.99</td>
      <td>10.050000</td>
      <td>-0.020000</td>
    </tr>
    <tr>
      <th>1815</th>
      <td>5867</td>
      <td>PJ</td>
      <td>Nassi</td>
      <td>5867</td>
      <td>USD</td>
      <td>32.16</td>
      <td>9.95</td>
      <td>42.11</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.49865E+13</td>
      <td>3.0</td>
      <td>1.441042e+09</td>
      <td>authorize.net</td>
      <td>42.11</td>
      <td>7483749716</td>
      <td>263228</td>
      <td>1.0</td>
      <td>1.441042e+09</td>
      <td>2015-09-02 20:32:37</td>
      <td>2015-08-31 19:35:17</td>
      <td>7614.0</td>
      <td>2110.0</td>
      <td>MoliCare Disposable Super Plus Briefs, Large/X...</td>
      <td>3.0</td>
      <td>10.72</td>
      <td>19.330000</td>
      <td>-2.870000</td>
    </tr>
    <tr>
      <th>2057</th>
      <td>7327</td>
      <td>Leigh Anne</td>
      <td>Duncan</td>
      <td>7327</td>
      <td>USD</td>
      <td>99.90</td>
      <td>0.00</td>
      <td>99.90</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.74823E+11</td>
      <td>3.0</td>
      <td>1.445780e+09</td>
      <td>paypal</td>
      <td>99.90</td>
      <td>1W093636Y5791203L</td>
      <td>02708Z</td>
      <td>1.0</td>
      <td>1.445780e+09</td>
      <td>2015-11-02 19:32:49</td>
      <td>2015-10-27 07:01:00</td>
      <td>9368.0</td>
      <td>1454.0</td>
      <td>Medline Sterile Bordered Gauze</td>
      <td>10.0</td>
      <td>9.99</td>
      <td>10.050000</td>
      <td>-0.006000</td>
    </tr>
    <tr>
      <th>2184</th>
      <td>8629</td>
      <td>Michael</td>
      <td>Zayats</td>
      <td>8629</td>
      <td>USD</td>
      <td>19.98</td>
      <td>5.54</td>
      <td>17.98</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>9.4055E+21</td>
      <td>3.0</td>
      <td>1.448205e+09</td>
      <td>authorize.net</td>
      <td>17.98</td>
      <td>7727592306</td>
      <td>513</td>
      <td>1.0</td>
      <td>1.448060e+09</td>
      <td>2015-11-26 10:48:07</td>
      <td>2015-11-23 09:09:28</td>
      <td>10719.0</td>
      <td>1454.0</td>
      <td>Sterile Bordered Gauze</td>
      <td>2.0</td>
      <td>9.99</td>
      <td>10.050000</td>
      <td>-0.030000</td>
    </tr>
    <tr>
      <th>2615</th>
      <td>11582</td>
      <td>Teresa</td>
      <td>Joslin</td>
      <td>11582</td>
      <td>USD</td>
      <td>65.99</td>
      <td>6.93</td>
      <td>65.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.75441E+11</td>
      <td>3.0</td>
      <td>1.452899e+09</td>
      <td>authorize.net</td>
      <td>65.99</td>
      <td>7897798685</td>
      <td>01591R</td>
      <td>1.0</td>
      <td>1.452899e+09</td>
      <td>2016-02-08 11:29:29</td>
      <td>2016-01-18 12:21:13</td>
      <td>13844.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>65.99</td>
      <td>88.500000</td>
      <td>-22.510000</td>
    </tr>
    <tr>
      <th>3071</th>
      <td>13228</td>
      <td>Barbara</td>
      <td>Hadley</td>
      <td>13228</td>
      <td>USD</td>
      <td>176.97</td>
      <td>23.67</td>
      <td>176.97</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.75731E+11</td>
      <td>3.0</td>
      <td>1.456368e+09</td>
      <td>authorize.net</td>
      <td>176.97</td>
      <td>8013360365</td>
      <td>243464</td>
      <td>1.0</td>
      <td>1.456368e+09</td>
      <td>2016-03-02 14:15:17</td>
      <td>2016-02-25 07:21:39</td>
      <td>15648.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>3.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-9.836667</td>
    </tr>
    <tr>
      <th>3086</th>
      <td>13267</td>
      <td>Marcy</td>
      <td>Seaman</td>
      <td>13267</td>
      <td>USD</td>
      <td>119.97</td>
      <td>0.00</td>
      <td>119.97</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.63949E+11</td>
      <td>3.0</td>
      <td>1.456436e+09</td>
      <td>authorize.net</td>
      <td>119.97</td>
      <td>8016026233</td>
      <td>245275</td>
      <td>1.0</td>
      <td>1.456436e+09</td>
      <td>2016-03-02 14:16:22</td>
      <td>2016-03-01 07:11:54</td>
      <td>15691.0</td>
      <td>17051.0</td>
      <td>Bottom Buddy toilet tissue aid</td>
      <td>3.0</td>
      <td>39.99</td>
      <td>40.500000</td>
      <td>-0.170000</td>
    </tr>
    <tr>
      <th>3095</th>
      <td>13302</td>
      <td>Melody</td>
      <td>Hollowell</td>
      <td>13302</td>
      <td>USD</td>
      <td>58.99</td>
      <td>6.71</td>
      <td>58.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.75743E+11</td>
      <td>3.0</td>
      <td>1.456506e+09</td>
      <td>authorize.net</td>
      <td>58.99</td>
      <td>8018282377</td>
      <td>10043</td>
      <td>1.0</td>
      <td>1.456506e+09</td>
      <td>2016-03-02 14:17:21</td>
      <td>2016-02-26 09:04:14</td>
      <td>15731.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-29.510000</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>13529</td>
      <td>Martha</td>
      <td>Keler</td>
      <td>13529</td>
      <td>USD</td>
      <td>69.99</td>
      <td>6.71</td>
      <td>69.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.75777E+11</td>
      <td>3.0</td>
      <td>1.456858e+09</td>
      <td>authorize.net</td>
      <td>69.99</td>
      <td>8031260682</td>
      <td>6135</td>
      <td>1.0</td>
      <td>1.456858e+09</td>
      <td>2016-03-02 14:23:42</td>
      <td>2016-03-02 08:49:41</td>
      <td>15970.0</td>
      <td>1453.0</td>
      <td>Sterile Bordered Gauze</td>
      <td>1.0</td>
      <td>69.99</td>
      <td>78.140000</td>
      <td>-8.150000</td>
    </tr>
    <tr>
      <th>3180</th>
      <td>16691</td>
      <td>Staci</td>
      <td>Meredith</td>
      <td>16691</td>
      <td>USD</td>
      <td>14.85</td>
      <td>0.00</td>
      <td>14.85</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.462487e+09</td>
      <td>authorize.net</td>
      <td>14.85</td>
      <td>8382059188</td>
      <td>10280</td>
      <td>5.0</td>
      <td>1.462487e+09</td>
      <td>2016-05-06 06:26:55</td>
      <td>2015-11-05 07:18:20</td>
      <td>19385.0</td>
      <td>1842.0</td>
      <td>Medline Emesis/Barf Bags, throw up bags, Blue ...</td>
      <td>1.0</td>
      <td>14.85</td>
      <td>15.500000</td>
      <td>-0.650000</td>
    </tr>
    <tr>
      <th>3206</th>
      <td>13832</td>
      <td>Debra</td>
      <td>Jordan</td>
      <td>13832</td>
      <td>USD</td>
      <td>58.99</td>
      <td>7.16</td>
      <td>58.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.75813E+11</td>
      <td>3.0</td>
      <td>1.457365e+09</td>
      <td>authorize.net</td>
      <td>58.99</td>
      <td>8048836159</td>
      <td>45498</td>
      <td>1.0</td>
      <td>1.457365e+09</td>
      <td>2016-03-07 11:23:36</td>
      <td>2016-03-07 11:23:36</td>
      <td>16293.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-29.510000</td>
    </tr>
    <tr>
      <th>3226</th>
      <td>13866</td>
      <td>CORI</td>
      <td>WALTER</td>
      <td>13866</td>
      <td>USD</td>
      <td>58.99</td>
      <td>7.85</td>
      <td>53.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.75822E+11</td>
      <td>3.0</td>
      <td>1.457401e+09</td>
      <td>authorize.net</td>
      <td>53.99</td>
      <td>8051104160</td>
      <td>1260</td>
      <td>1.0</td>
      <td>1.457401e+09</td>
      <td>2016-03-08 09:26:25</td>
      <td>2016-03-08 09:26:25</td>
      <td>16328.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-29.510000</td>
    </tr>
    <tr>
      <th>3374</th>
      <td>14351</td>
      <td>Ann</td>
      <td>Orlando</td>
      <td>14351</td>
      <td>USD</td>
      <td>27.99</td>
      <td>0.00</td>
      <td>27.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>4.83833E+14</td>
      <td>3.0</td>
      <td>1.458577e+09</td>
      <td>authorize.net</td>
      <td>27.99</td>
      <td>8090515241</td>
      <td>56410</td>
      <td>1.0</td>
      <td>1.458577e+09</td>
      <td>2016-03-21 15:17:30</td>
      <td>2016-03-21 14:13:14</td>
      <td>16855.0</td>
      <td>782.0</td>
      <td>Bed Assist Bar with Storage Pocket</td>
      <td>1.0</td>
      <td>27.99</td>
      <td>29.160000</td>
      <td>-1.170000</td>
    </tr>
    <tr>
      <th>3398</th>
      <td>14548</td>
      <td>Bob</td>
      <td>Wiright</td>
      <td>14548</td>
      <td>USD</td>
      <td>34.99</td>
      <td>9.95</td>
      <td>53.42</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.63949E+11</td>
      <td>3.0</td>
      <td>1.458757e+09</td>
      <td>authorize.net</td>
      <td>53.42</td>
      <td>8097412082</td>
      <td>09877C</td>
      <td>1.0</td>
      <td>1.458757e+09</td>
      <td>2016-03-29 18:33:03</td>
      <td>2016-03-29 18:33:03</td>
      <td>17069.0</td>
      <td>17051.0</td>
      <td>Bottom Buddy toilet tissue aid</td>
      <td>1.0</td>
      <td>34.99</td>
      <td>40.500000</td>
      <td>-5.510000</td>
    </tr>
    <tr>
      <th>3405</th>
      <td>14597</td>
      <td>Alouis</td>
      <td>Colgan</td>
      <td>14597</td>
      <td>USD</td>
      <td>34.99</td>
      <td>0.00</td>
      <td>34.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.63949E+11</td>
      <td>3.0</td>
      <td>1.458804e+09</td>
      <td>authorize.net</td>
      <td>34.99</td>
      <td>8099072722</td>
      <td>161248</td>
      <td>1.0</td>
      <td>1.458804e+09</td>
      <td>2016-03-31 07:05:07</td>
      <td>2016-03-31 07:05:07</td>
      <td>17119.0</td>
      <td>17051.0</td>
      <td>Bottom Buddy toilet tissue aid</td>
      <td>1.0</td>
      <td>34.99</td>
      <td>40.500000</td>
      <td>-5.510000</td>
    </tr>
    <tr>
      <th>3409</th>
      <td>14619</td>
      <td>lois</td>
      <td>moore</td>
      <td>14619</td>
      <td>USD</td>
      <td>34.99</td>
      <td>0.00</td>
      <td>34.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.63949E+11</td>
      <td>3.0</td>
      <td>1.458838e+09</td>
      <td>authorize.net</td>
      <td>34.99</td>
      <td>8100164574</td>
      <td>24024</td>
      <td>1.0</td>
      <td>1.458838e+09</td>
      <td>2016-03-29 18:38:57</td>
      <td>2016-03-29 18:38:57</td>
      <td>17141.0</td>
      <td>17051.0</td>
      <td>Bottom Buddy toilet tissue aid</td>
      <td>1.0</td>
      <td>34.99</td>
      <td>40.500000</td>
      <td>-5.510000</td>
    </tr>
    <tr>
      <th>3420</th>
      <td>14710</td>
      <td>Sherman</td>
      <td>Langer</td>
      <td>14710</td>
      <td>USD</td>
      <td>58.99</td>
      <td>0.00</td>
      <td>58.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.7597E+11</td>
      <td>3.0</td>
      <td>1.459004e+09</td>
      <td>authorize.net</td>
      <td>58.99</td>
      <td>8105679785</td>
      <td>01196D</td>
      <td>1.0</td>
      <td>1.459004e+09</td>
      <td>2016-03-28 06:49:18</td>
      <td>2016-03-28 06:49:18</td>
      <td>17234.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-29.510000</td>
    </tr>
    <tr>
      <th>3487</th>
      <td>14953</td>
      <td>Michael</td>
      <td>Ball</td>
      <td>14953</td>
      <td>USD</td>
      <td>58.99</td>
      <td>0.00</td>
      <td>58.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.13E+19</td>
      <td>3.0</td>
      <td>1.459381e+09</td>
      <td>authorize.net</td>
      <td>58.99</td>
      <td>8117784458</td>
      <td>322162</td>
      <td>1.0</td>
      <td>1.459381e+09</td>
      <td>2016-03-30 18:49:32</td>
      <td>2016-03-30 18:49:32</td>
      <td>17496.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-29.510000</td>
    </tr>
    <tr>
      <th>3541</th>
      <td>15144</td>
      <td>Janice</td>
      <td>Boyle</td>
      <td>15144</td>
      <td>USD</td>
      <td>27.99</td>
      <td>0.00</td>
      <td>27.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.459802e+09</td>
      <td>authorize.net</td>
      <td>27.99</td>
      <td>8133377121</td>
      <td>08674Z</td>
      <td>5.0</td>
      <td>1.459802e+09</td>
      <td>2016-04-04 15:36:45</td>
      <td>2015-11-05 07:18:20</td>
      <td>17709.0</td>
      <td>782.0</td>
      <td>Bed Assist Bar with Storage Pocket</td>
      <td>1.0</td>
      <td>27.99</td>
      <td>29.160000</td>
      <td>-1.170000</td>
    </tr>
    <tr>
      <th>3542</th>
      <td>15145</td>
      <td>Janice</td>
      <td>Boyle</td>
      <td>15145</td>
      <td>USD</td>
      <td>27.99</td>
      <td>0.00</td>
      <td>25.19</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.459803e+09</td>
      <td>authorize.net</td>
      <td>25.19</td>
      <td>8133402033</td>
      <td>08307Z</td>
      <td>4.0</td>
      <td>1.459803e+09</td>
      <td>2016-04-04 14:09:07</td>
      <td>2015-11-05 07:18:20</td>
      <td>17710.0</td>
      <td>782.0</td>
      <td>Bed Assist Bar with Storage Pocket</td>
      <td>1.0</td>
      <td>27.99</td>
      <td>29.160000</td>
      <td>-1.170000</td>
    </tr>
    <tr>
      <th>3583</th>
      <td>15260</td>
      <td>Wesley</td>
      <td>Chalker Jr</td>
      <td>15260</td>
      <td>USD</td>
      <td>19.98</td>
      <td>0.00</td>
      <td>19.98</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>4.83833E+14</td>
      <td>3.0</td>
      <td>1.460044e+09</td>
      <td>authorize.net</td>
      <td>19.98</td>
      <td>8142281323</td>
      <td>7994</td>
      <td>1.0</td>
      <td>1.460044e+09</td>
      <td>2016-04-11 13:23:14</td>
      <td>2016-04-11 13:23:14</td>
      <td>17832.0</td>
      <td>1454.0</td>
      <td>Sterile Bordered Gauze</td>
      <td>2.0</td>
      <td>9.99</td>
      <td>10.050000</td>
      <td>-0.030000</td>
    </tr>
    <tr>
      <th>3584</th>
      <td>16364</td>
      <td>Wesley</td>
      <td>Chalker Jr</td>
      <td>16364</td>
      <td>USD</td>
      <td>19.98</td>
      <td>0.00</td>
      <td>19.98</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>9.4055E+21</td>
      <td>3.0</td>
      <td>1.462125e+09</td>
      <td>authorize.net</td>
      <td>19.98</td>
      <td>8366965020</td>
      <td>1079</td>
      <td>1.0</td>
      <td>1.462125e+09</td>
      <td>2016-05-02 11:44:04</td>
      <td>2016-05-02 11:44:04</td>
      <td>19040.0</td>
      <td>1454.0</td>
      <td>Sterile Bordered Gauze</td>
      <td>2.0</td>
      <td>9.99</td>
      <td>10.050000</td>
      <td>-0.030000</td>
    </tr>
    <tr>
      <th>3615</th>
      <td>15376</td>
      <td>CLARENCE</td>
      <td>POLLARD</td>
      <td>15376</td>
      <td>USD</td>
      <td>55.98</td>
      <td>0.00</td>
      <td>50.98</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>1.49935E+13</td>
      <td>3.0</td>
      <td>1.460245e+09</td>
      <td>paypal</td>
      <td>50.98</td>
      <td>7WL48849TP228631L</td>
      <td>02708Z</td>
      <td>1.0</td>
      <td>1.460245e+09</td>
      <td>2016-04-12 08:56:07</td>
      <td>2016-04-12 08:56:07</td>
      <td>17962.0</td>
      <td>782.0</td>
      <td>Bed Assist Bar with Storage Pocket</td>
      <td>2.0</td>
      <td>27.99</td>
      <td>29.160000</td>
      <td>-0.585000</td>
    </tr>
    <tr>
      <th>3618</th>
      <td>15393</td>
      <td>Barbara</td>
      <td>James</td>
      <td>15393</td>
      <td>USD</td>
      <td>58.99</td>
      <td>0.00</td>
      <td>58.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.76076E+11</td>
      <td>3.0</td>
      <td>1.460294e+09</td>
      <td>authorize.net</td>
      <td>58.99</td>
      <td>8303422727</td>
      <td>82115</td>
      <td>1.0</td>
      <td>1.460294e+09</td>
      <td>2016-04-13 11:04:41</td>
      <td>2016-04-11 12:01:51</td>
      <td>17981.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-29.510000</td>
    </tr>
    <tr>
      <th>3619</th>
      <td>15394</td>
      <td>WARREN</td>
      <td>SEDRAN</td>
      <td>15394</td>
      <td>USD</td>
      <td>27.99</td>
      <td>0.00</td>
      <td>25.19</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>4.83833E+14</td>
      <td>3.0</td>
      <td>1.460297e+09</td>
      <td>authorize.net</td>
      <td>25.19</td>
      <td>8303477615</td>
      <td>600479</td>
      <td>1.0</td>
      <td>1.460297e+09</td>
      <td>2016-04-12 08:58:35</td>
      <td>2016-04-12 08:58:35</td>
      <td>17982.0</td>
      <td>782.0</td>
      <td>Bed Assist Bar with Storage Pocket</td>
      <td>1.0</td>
      <td>27.99</td>
      <td>29.160000</td>
      <td>-1.170000</td>
    </tr>
    <tr>
      <th>3719</th>
      <td>15780</td>
      <td>Lenny</td>
      <td>Shenall</td>
      <td>15780</td>
      <td>USD</td>
      <td>58.99</td>
      <td>9.95</td>
      <td>64.08</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.76132E+11</td>
      <td>3.0</td>
      <td>1.461005e+09</td>
      <td>authorize.net</td>
      <td>64.08</td>
      <td>8328252812</td>
      <td>21864</td>
      <td>1.0</td>
      <td>1.461005e+09</td>
      <td>2016-04-18 11:57:28</td>
      <td>2016-04-18 11:57:28</td>
      <td>18396.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-29.510000</td>
    </tr>
    <tr>
      <th>3731</th>
      <td>15844</td>
      <td>Dolores</td>
      <td>Gosnell</td>
      <td>15844</td>
      <td>USD</td>
      <td>27.99</td>
      <td>9.95</td>
      <td>27.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>9.53516E+14</td>
      <td>3.0</td>
      <td>1.461098e+09</td>
      <td>authorize.net</td>
      <td>27.99</td>
      <td>8332005134</td>
      <td>01990R</td>
      <td>1.0</td>
      <td>1.461098e+09</td>
      <td>2016-05-04 11:36:46</td>
      <td>2016-05-04 11:36:46</td>
      <td>18462.0</td>
      <td>782.0</td>
      <td>Medline Bed Assist Bar</td>
      <td>1.0</td>
      <td>27.99</td>
      <td>29.160000</td>
      <td>-1.170000</td>
    </tr>
    <tr>
      <th>3752</th>
      <td>15899</td>
      <td>Lee</td>
      <td>Gerstenhaber</td>
      <td>15899</td>
      <td>USD</td>
      <td>34.99</td>
      <td>0.00</td>
      <td>34.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>9.4055E+21</td>
      <td>3.0</td>
      <td>1.461190e+09</td>
      <td>paypal</td>
      <td>34.99</td>
      <td>2HB74261TB057390A</td>
      <td>02708Z</td>
      <td>1.0</td>
      <td>1.461190e+09</td>
      <td>2016-04-21 10:51:00</td>
      <td>2016-04-21 10:51:00</td>
      <td>18519.0</td>
      <td>17051.0</td>
      <td>Bottom Buddy toilet tissue aid</td>
      <td>1.0</td>
      <td>34.99</td>
      <td>40.500000</td>
      <td>-5.510000</td>
    </tr>
    <tr>
      <th>3766</th>
      <td>15943</td>
      <td>Jan</td>
      <td>Kripzer</td>
      <td>15943</td>
      <td>USD</td>
      <td>34.99</td>
      <td>9.95</td>
      <td>34.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>9.4055E+19</td>
      <td>3.0</td>
      <td>1.461274e+09</td>
      <td>authorize.net</td>
      <td>34.99</td>
      <td>8338712178</td>
      <td>06826D</td>
      <td>1.0</td>
      <td>1.461274e+09</td>
      <td>2016-04-21 14:27:09</td>
      <td>2016-04-21 14:27:09</td>
      <td>18566.0</td>
      <td>17051.0</td>
      <td>Bottom Buddy toilet tissue aid</td>
      <td>1.0</td>
      <td>34.99</td>
      <td>40.500000</td>
      <td>-5.510000</td>
    </tr>
    <tr>
      <th>3774</th>
      <td>15970</td>
      <td>Christopher</td>
      <td>Zanini</td>
      <td>15970</td>
      <td>USD</td>
      <td>27.99</td>
      <td>0.00</td>
      <td>25.19</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>1.02633E+14</td>
      <td>3.0</td>
      <td>1.461604e+09</td>
      <td>authorize.net</td>
      <td>25.19</td>
      <td>8341795724</td>
      <td>275057</td>
      <td>1.0</td>
      <td>1.461356e+09</td>
      <td>2016-04-25 19:21:18</td>
      <td>2016-04-25 18:19:10</td>
      <td>18593.0</td>
      <td>782.0</td>
      <td>Medline Bed Assist Bar</td>
      <td>1.0</td>
      <td>27.99</td>
      <td>29.160000</td>
      <td>-1.170000</td>
    </tr>
    <tr>
      <th>3779</th>
      <td>15992</td>
      <td>Gail</td>
      <td>Reel</td>
      <td>15992</td>
      <td>USD</td>
      <td>34.99</td>
      <td>0.00</td>
      <td>34.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>9.4055E+21</td>
      <td>3.0</td>
      <td>1.461605e+09</td>
      <td>authorize.net</td>
      <td>34.99</td>
      <td>8343113413</td>
      <td>57116</td>
      <td>1.0</td>
      <td>1.461407e+09</td>
      <td>2016-04-25 10:25:23</td>
      <td>2016-04-25 10:25:23</td>
      <td>18616.0</td>
      <td>17051.0</td>
      <td>Bottom Buddy toilet tissue aid</td>
      <td>1.0</td>
      <td>34.99</td>
      <td>40.500000</td>
      <td>-5.510000</td>
    </tr>
    <tr>
      <th>3789</th>
      <td>16005</td>
      <td>ANNE MARIE</td>
      <td>DODERO</td>
      <td>16005</td>
      <td>USD</td>
      <td>27.99</td>
      <td>0.00</td>
      <td>27.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.96468E+14</td>
      <td>3.0</td>
      <td>1.461604e+09</td>
      <td>authorize.net</td>
      <td>27.99</td>
      <td>8344004179</td>
      <td>01862A</td>
      <td>1.0</td>
      <td>1.461441e+09</td>
      <td>2016-04-25 18:51:06</td>
      <td>2016-04-25 17:47:06</td>
      <td>18637.0</td>
      <td>782.0</td>
      <td>Medline Bed Assist Bar</td>
      <td>1.0</td>
      <td>27.99</td>
      <td>29.160000</td>
      <td>-1.170000</td>
    </tr>
    <tr>
      <th>3813</th>
      <td>16068</td>
      <td>betty</td>
      <td>montesi</td>
      <td>16068</td>
      <td>USD</td>
      <td>58.99</td>
      <td>0.00</td>
      <td>58.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.76208E+11</td>
      <td>3.0</td>
      <td>1.461782e+09</td>
      <td>authorize.net</td>
      <td>58.99</td>
      <td>8347372581</td>
      <td>09654D</td>
      <td>1.0</td>
      <td>1.461593e+09</td>
      <td>2016-04-27 11:29:54</td>
      <td>2016-04-27 11:29:54</td>
      <td>18704.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-29.510000</td>
    </tr>
    <tr>
      <th>8237</th>
      <td>16204</td>
      <td>Jill</td>
      <td>OGorman</td>
      <td>16204</td>
      <td>USD</td>
      <td>34.99</td>
      <td>0.00</td>
      <td>34.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>9.4055E+21</td>
      <td>3.0</td>
      <td>1.461881e+09</td>
      <td>authorize.net</td>
      <td>34.99</td>
      <td>8356224503</td>
      <td>825606</td>
      <td>1.0</td>
      <td>1.461810e+09</td>
      <td>2016-05-02 11:29:47</td>
      <td>2016-05-02 11:29:47</td>
      <td>18863.0</td>
      <td>17051.0</td>
      <td>Bottom Buddy toilet tissue aid</td>
      <td>1.0</td>
      <td>34.99</td>
      <td>40.500000</td>
      <td>-5.510000</td>
    </tr>
    <tr>
      <th>8274</th>
      <td>16335</td>
      <td>Casandra</td>
      <td>McMorries</td>
      <td>16335</td>
      <td>USD</td>
      <td>58.99</td>
      <td>0.00</td>
      <td>58.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.76248E+11</td>
      <td>3.0</td>
      <td>1.462133e+09</td>
      <td>authorize.net</td>
      <td>58.99</td>
      <td>8364716076</td>
      <td>19481</td>
      <td>1.0</td>
      <td>1.462054e+09</td>
      <td>2016-05-03 10:46:39</td>
      <td>2016-05-03 10:46:39</td>
      <td>19010.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-29.510000</td>
    </tr>
    <tr>
      <th>8307</th>
      <td>16414</td>
      <td>Douglas</td>
      <td>Monsoor</td>
      <td>16414</td>
      <td>USD</td>
      <td>104.97</td>
      <td>9.95</td>
      <td>104.97</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>7.76253E+11</td>
      <td>3.0</td>
      <td>1.462217e+09</td>
      <td>authorize.net</td>
      <td>104.97</td>
      <td>8370811515</td>
      <td>01110C</td>
      <td>1.0</td>
      <td>1.462217e+09</td>
      <td>2016-05-03 11:58:54</td>
      <td>2016-05-03 11:58:54</td>
      <td>19092.0</td>
      <td>17051.0</td>
      <td>Bottom Buddy toilet tissue aid</td>
      <td>3.0</td>
      <td>34.99</td>
      <td>40.500000</td>
      <td>-1.836667</td>
    </tr>
    <tr>
      <th>8353</th>
      <td>16541</td>
      <td>Lester</td>
      <td>Curnow</td>
      <td>16541</td>
      <td>USD</td>
      <td>14.85</td>
      <td>0.00</td>
      <td>14.85</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>9.36129E+21</td>
      <td>3.0</td>
      <td>1.462373e+09</td>
      <td>authorize.net</td>
      <td>14.85</td>
      <td>8376930603</td>
      <td>286988</td>
      <td>1.0</td>
      <td>1.462373e+09</td>
      <td>2016-05-05 09:21:11</td>
      <td>2016-05-05 09:21:11</td>
      <td>19227.0</td>
      <td>1842.0</td>
      <td>Medline Emesis/Barf Bags, throw up bags, Blue ...</td>
      <td>1.0</td>
      <td>14.85</td>
      <td>15.500000</td>
      <td>-0.650000</td>
    </tr>
    <tr>
      <th>8359</th>
      <td>16558</td>
      <td>Tom</td>
      <td>Beatty</td>
      <td>16558</td>
      <td>USD</td>
      <td>27.99</td>
      <td>0.00</td>
      <td>27.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.462391e+09</td>
      <td>authorize.net</td>
      <td>27.99</td>
      <td>8378075808</td>
      <td>05297C</td>
      <td>4.0</td>
      <td>1.462391e+09</td>
      <td>2016-05-04 12:37:07</td>
      <td>2015-11-05 07:18:20</td>
      <td>19245.0</td>
      <td>782.0</td>
      <td>Medline Bed Assist Bar</td>
      <td>1.0</td>
      <td>27.99</td>
      <td>28.160000</td>
      <td>-0.170000</td>
    </tr>
    <tr>
      <th>8365</th>
      <td>16597</td>
      <td>Patricia</td>
      <td>Kotsenas</td>
      <td>16597</td>
      <td>USD</td>
      <td>64.80</td>
      <td>9.95</td>
      <td>74.75</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>2.27196E+13</td>
      <td>3.0</td>
      <td>1.462409e+09</td>
      <td>authorize.net</td>
      <td>74.75</td>
      <td>8378981908</td>
      <td>02630B</td>
      <td>1.0</td>
      <td>1.462409e+09</td>
      <td>2016-05-05 11:11:21</td>
      <td>2016-05-05 10:07:32</td>
      <td>19285.0</td>
      <td>1025.0</td>
      <td>Steel Bariatric Commode</td>
      <td>1.0</td>
      <td>64.80</td>
      <td>92.000000</td>
      <td>-27.200000</td>
    </tr>
    <tr>
      <th>8384</th>
      <td>16693</td>
      <td>Rebecca</td>
      <td>Griffiths</td>
      <td>16693</td>
      <td>USD</td>
      <td>58.99</td>
      <td>0.00</td>
      <td>58.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>575-642-8976</td>
      <td>3.0</td>
      <td>1.462488e+09</td>
      <td>authorize.net</td>
      <td>58.99</td>
      <td>8382102271</td>
      <td>586726</td>
      <td>1.0</td>
      <td>1.462488e+09</td>
      <td>2016-05-06 06:31:57</td>
      <td>2016-05-06 06:31:57</td>
      <td>19387.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-29.510000</td>
    </tr>
    <tr>
      <th>8392</th>
      <td>16744</td>
      <td>June</td>
      <td>Rudner</td>
      <td>16744</td>
      <td>USD</td>
      <td>27.99</td>
      <td>0.00</td>
      <td>27.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>4.8383E+14</td>
      <td>3.0</td>
      <td>1.462540e+09</td>
      <td>authorize.net</td>
      <td>27.99</td>
      <td>8383400273</td>
      <td>00615Z</td>
      <td>1.0</td>
      <td>1.462540e+09</td>
      <td>2016-05-06 14:47:12</td>
      <td>2016-05-06 13:43:12</td>
      <td>19439.0</td>
      <td>782.0</td>
      <td>Medline Bed Assist Bar</td>
      <td>1.0</td>
      <td>27.99</td>
      <td>28.160000</td>
      <td>-0.170000</td>
    </tr>
    <tr>
      <th>8394</th>
      <td>16746</td>
      <td>FRANK</td>
      <td>SOPINSKI</td>
      <td>16746</td>
      <td>USD</td>
      <td>27.99</td>
      <td>0.00</td>
      <td>27.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.40791E+13</td>
      <td>3.0</td>
      <td>1.462543e+09</td>
      <td>authorize.net</td>
      <td>27.99</td>
      <td>8383552971</td>
      <td>06375C</td>
      <td>1.0</td>
      <td>1.462543e+09</td>
      <td>2016-05-06 16:47:02</td>
      <td>2016-05-06 15:40:44</td>
      <td>19441.0</td>
      <td>782.0</td>
      <td>Medline Bed Assist Bar</td>
      <td>1.0</td>
      <td>27.99</td>
      <td>28.160000</td>
      <td>-0.170000</td>
    </tr>
    <tr>
      <th>8395</th>
      <td>16747</td>
      <td>michael</td>
      <td>shotts</td>
      <td>16747</td>
      <td>USD</td>
      <td>74.99</td>
      <td>9.95</td>
      <td>77.44</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>8.3565E+14</td>
      <td>3.0</td>
      <td>1.462544e+09</td>
      <td>authorize.net</td>
      <td>77.44</td>
      <td>8383594730</td>
      <td>18242</td>
      <td>1.0</td>
      <td>1.462544e+09</td>
      <td>2016-05-06 13:17:19</td>
      <td>2016-05-06 12:14:14</td>
      <td>19442.0</td>
      <td>426.0</td>
      <td>Padded Transfer Benches</td>
      <td>1.0</td>
      <td>74.99</td>
      <td>77.000000</td>
      <td>-2.010000</td>
    </tr>
    <tr>
      <th>8396</th>
      <td>16749</td>
      <td>Holly</td>
      <td>Scotchel</td>
      <td>16749</td>
      <td>USD</td>
      <td>58.99</td>
      <td>0.00</td>
      <td>58.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.462545e+09</td>
      <td>authorize.net</td>
      <td>58.99</td>
      <td>8383648902</td>
      <td>01683C</td>
      <td>5.0</td>
      <td>1.462545e+09</td>
      <td>2016-05-06 07:39:00</td>
      <td>2015-11-05 07:18:20</td>
      <td>19444.0</td>
      <td>1841.0</td>
      <td>Emesis Bags, Blue, 36.000 OZ</td>
      <td>1.0</td>
      <td>58.99</td>
      <td>88.500000</td>
      <td>-29.510000</td>
    </tr>
    <tr>
      <th>8402</th>
      <td>16753</td>
      <td>Harold</td>
      <td>Roberts</td>
      <td>16753</td>
      <td>USD</td>
      <td>27.99</td>
      <td>0.00</td>
      <td>27.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>8.3565E+14</td>
      <td>3.0</td>
      <td>1.462546e+09</td>
      <td>authorize.net</td>
      <td>27.99</td>
      <td>8383741325</td>
      <td>5242</td>
      <td>1.0</td>
      <td>1.462546e+09</td>
      <td>2016-05-06 15:18:01</td>
      <td>2016-05-06 14:13:22</td>
      <td>19452.0</td>
      <td>782.0</td>
      <td>Medline Bed Assist Bar</td>
      <td>1.0</td>
      <td>27.99</td>
      <td>28.160000</td>
      <td>-0.170000</td>
    </tr>
    <tr>
      <th>8428</th>
      <td>16962</td>
      <td>Arthur</td>
      <td>Royer</td>
      <td>16962</td>
      <td>USD</td>
      <td>44.00</td>
      <td>9.95</td>
      <td>53.95</td>
      <td>manual</td>
      <td>0|Standard Shipping</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.462804e+09</td>
      <td>authorize.net</td>
      <td>24.95</td>
      <td>8390592894</td>
      <td>909003</td>
      <td>5.0</td>
      <td>1.462804e+09</td>
      <td>2016-05-09 12:56:09</td>
      <td>2016-05-09 12:55:41</td>
      <td>19674.0</td>
      <td>1842.0</td>
      <td>BUCKET, REPLACEMENT FOR MDS89668XW</td>
      <td>1.0</td>
      <td>44.00</td>
      <td>51.112718</td>
      <td>-7.112718</td>
    </tr>
    <tr>
      <th>8434</th>
      <td>16981</td>
      <td>Rebecca</td>
      <td>Schnepf</td>
      <td>16981</td>
      <td>USD</td>
      <td>34.99</td>
      <td>0.00</td>
      <td>34.99</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.462823e+09</td>
      <td>authorize.net</td>
      <td>34.99</td>
      <td>8391980366</td>
      <td>43453</td>
      <td>0.0</td>
      <td>1.462823e+09</td>
      <td>2016-05-10 14:08:06</td>
      <td>2015-11-05 07:18:20</td>
      <td>19696.0</td>
      <td>17051.0</td>
      <td>Bottom Buddy toilet tissue aid</td>
      <td>1.0</td>
      <td>34.99</td>
      <td>40.500000</td>
      <td>-5.510000</td>
    </tr>
    <tr>
      <th>8435</th>
      <td>16987</td>
      <td>Patricia</td>
      <td>Sleeper</td>
      <td>16987</td>
      <td>USD</td>
      <td>14.85</td>
      <td>0.00</td>
      <td>14.85</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.462824e+09</td>
      <td>authorize.net</td>
      <td>14.85</td>
      <td>8392053758</td>
      <td>609572</td>
      <td>5.0</td>
      <td>1.462824e+09</td>
      <td>2016-05-09 13:39:33</td>
      <td>2015-11-05 07:18:20</td>
      <td>19703.0</td>
      <td>1842.0</td>
      <td>Medline Emesis/Barf Bags, throw up bags, Blue ...</td>
      <td>1.0</td>
      <td>14.85</td>
      <td>15.500000</td>
      <td>-0.650000</td>
    </tr>
    <tr>
      <th>8448</th>
      <td>17060</td>
      <td>Nick</td>
      <td>Dziurkowski</td>
      <td>17060</td>
      <td>USD</td>
      <td>14.85</td>
      <td>0.00</td>
      <td>14.85</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.462907e+09</td>
      <td>authorize.net</td>
      <td>14.85</td>
      <td>8395155339</td>
      <td>13223</td>
      <td>5.0</td>
      <td>1.462907e+09</td>
      <td>2016-05-13 07:51:23</td>
      <td>2015-11-05 07:18:20</td>
      <td>19782.0</td>
      <td>1842.0</td>
      <td>Medline Emesis/Barf Bags, throw up bags, Blue ...</td>
      <td>1.0</td>
      <td>14.85</td>
      <td>15.500000</td>
      <td>-0.650000</td>
    </tr>
    <tr>
      <th>8456</th>
      <td>17071</td>
      <td>Marvin</td>
      <td>Rahman</td>
      <td>17071</td>
      <td>USD</td>
      <td>14.85</td>
      <td>0.00</td>
      <td>14.85</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.462911e+09</td>
      <td>authorize.net</td>
      <td>14.85</td>
      <td>8395421530</td>
      <td>157796</td>
      <td>5.0</td>
      <td>1.462911e+09</td>
      <td>2016-05-13 07:54:32</td>
      <td>2015-11-05 07:18:20</td>
      <td>19798.0</td>
      <td>1842.0</td>
      <td>Medline Emesis/Barf Bags, throw up bags, Blue ...</td>
      <td>1.0</td>
      <td>14.85</td>
      <td>15.500000</td>
      <td>-0.650000</td>
    </tr>
    <tr>
      <th>8457</th>
      <td>17072</td>
      <td>Bethel</td>
      <td>Jones</td>
      <td>17072</td>
      <td>USD</td>
      <td>14.85</td>
      <td>9.95</td>
      <td>14.85</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.462912e+09</td>
      <td>authorize.net</td>
      <td>14.85</td>
      <td>8395506459</td>
      <td>01038R</td>
      <td>5.0</td>
      <td>1.462912e+09</td>
      <td>2016-05-13 07:58:05</td>
      <td>2015-11-05 07:18:20</td>
      <td>19799.0</td>
      <td>1842.0</td>
      <td>Medline Emesis/Barf Bags, throw up bags, Blue ...</td>
      <td>1.0</td>
      <td>14.85</td>
      <td>15.500000</td>
      <td>-0.650000</td>
    </tr>
    <tr>
      <th>8470</th>
      <td>17115</td>
      <td>Barbara</td>
      <td>Hawkins</td>
      <td>17115</td>
      <td>USD</td>
      <td>34.99</td>
      <td>0.00</td>
      <td>31.49</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>9.4055E+21</td>
      <td>3.0</td>
      <td>1.462941e+09</td>
      <td>authorize.net</td>
      <td>31.49</td>
      <td>8396533738</td>
      <td>10541</td>
      <td>1.0</td>
      <td>1.462941e+09</td>
      <td>2016-05-11 06:59:29</td>
      <td>2016-05-11 06:59:29</td>
      <td>19846.0</td>
      <td>17051.0</td>
      <td>Bottom Buddy toilet tissue aid</td>
      <td>1.0</td>
      <td>34.99</td>
      <td>40.500000</td>
      <td>-5.510000</td>
    </tr>
    <tr>
      <th>8504</th>
      <td>17206</td>
      <td>Gail</td>
      <td>Yarbrough</td>
      <td>17206</td>
      <td>USD</td>
      <td>14.85</td>
      <td>0.00</td>
      <td>14.85</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.463064e+09</td>
      <td>authorize.net</td>
      <td>14.85</td>
      <td>8400367859</td>
      <td>601427</td>
      <td>5.0</td>
      <td>1.463064e+09</td>
      <td>2016-05-13 08:00:03</td>
      <td>2015-11-05 07:18:20</td>
      <td>19948.0</td>
      <td>1842.0</td>
      <td>Medline Emesis/Barf Bags, throw up bags, Blue ...</td>
      <td>1.0</td>
      <td>14.85</td>
      <td>15.500000</td>
      <td>-0.650000</td>
    </tr>
    <tr>
      <th>8522</th>
      <td>17271</td>
      <td>Julia</td>
      <td>Gearhart</td>
      <td>17271</td>
      <td>USD</td>
      <td>29.70</td>
      <td>0.00</td>
      <td>29.70</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.463160e+09</td>
      <td>authorize.net</td>
      <td>29.70</td>
      <td>8404223306</td>
      <td>289728</td>
      <td>0.0</td>
      <td>1.463160e+09</td>
      <td>2015-08-06 06:40:10</td>
      <td>2015-11-05 07:18:20</td>
      <td>20020.0</td>
      <td>1842.0</td>
      <td>Medline Emesis/Barf Bags, throw up bags, Blue ...</td>
      <td>2.0</td>
      <td>14.85</td>
      <td>15.500000</td>
      <td>-0.325000</td>
    </tr>
    <tr>
      <th>8567</th>
      <td>17421</td>
      <td>Nora</td>
      <td>Fontana</td>
      <td>17421</td>
      <td>USD</td>
      <td>14.85</td>
      <td>0.00</td>
      <td>16.06</td>
      <td>fedex</td>
      <td>11|Ground</td>
      <td>6.87398E+14</td>
      <td>3.0</td>
      <td>1.463409e+09</td>
      <td>paypal</td>
      <td>16.06</td>
      <td>16706988SV3261147</td>
      <td>02708Z</td>
      <td>0.0</td>
      <td>1.463409e+09</td>
      <td>2015-08-06 06:40:10</td>
      <td>2015-11-05 07:18:20</td>
      <td>20186.0</td>
      <td>1842.0</td>
      <td>Medline Emesis/Barf Bags, throw up bags, Blue ...</td>
      <td>1.0</td>
      <td>14.85</td>
      <td>15.500000</td>
      <td>-0.650000</td>
    </tr>
  </tbody>
</table>
</div>
</details>

We can drill down and look at things like what shipping methods we use most. Maybe we can cut a deal with USPS or whoever we're doing the most business with. We can also see how much free shipping is happening, that may effect our bottom line.

```python
orders['shipping_method'].value_counts()
```

11|Ground                    2549
0|Standard Shipping           697
-1|Free Shipping              220
0|Free Shipping                86
NATIONAL DELIVERY               2
HOWARD'S EXPRESS, INC.          2
WILSON TRUCKING CORP            2
NATIONAL DELIVERY SYSTEMS       1
MEDTRANS                        1
SOUTHWESTERN MOTOR TRNAS        1
ROADWAY EXPRESS                 1
FEDERAL EXPRESS CORP.           1
INTERNATIONAL FEDEX             1
LAND AIR OF NEW ENGLAND         1
Name: shipping_method, dtype: int64

Now that this table is all cleaned up, we can move on to the final table.

## Products Table

As before we'll drop the duplicates, the null values (4 rows this time) and then do the feature transformations.

```python
products.drop_duplicates(subset= 'id', inplace = True)
products.dropna(inplace = True)

#Feature Transformations
products['last_modified'] = products['last_modified'].transform(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
products['import_id'] = products['import_id'].astype(str)
products['vendor'] = products['vendor'].astype(str)
```

At this point, we can use NLP to analyze our products, see which ones are most popular, and group them into the types of products that give us the most profit. Again, this is beyond the scope of this specific project, which is to create a basic customer segmentation and a RFM table (eyes on the prize).

# Writing to SQL Database

In this basic example, I will be using SQLite3, although I could use SQLalchemy or one of the other million packages that interface between python and SQL.

```python
#Create Connection
conn = sqlite3.connect('SQL Server Filepath\\.db')

#Customers table
customers.to_sql('customers', conn, if_exists='replace', index=True)

#Orders table
orders.to_sql('orders', conn, if_exists='replace', index=True)

#Products table
products.to_sql('products', conn, if_exists='replace', index=True)

conn.close()
```

# Groupby Analysis

With our first task behind us, we can start to see the finish line in the distance.... A bit far in the distance, but we can see it, and the fun bit is still ahead! Since we didn't really do a deep dive into the products table, we'll mostly be looking at our customer and order information.

```python
#Make a FullName variable to merge on easily
#People can share the same first or last name, but rarely share both
customers['FullName'] = customers['fname'] + customers['lname']
orders['FullName'] = orders['fname'] + orders['lname']

#Create the new DataFrame
df2 = pd.merge(customers, orders, how= 'outer', on='FullName')

#Drop the redundant columns
df2.drop(['fname_x', 'lname_x', 'fname_y', 'lname_y'], axis = 1, inplace = True)
```

Lets take another snapshot of our customer buying habits, similar to before, but now we have the profit column as well.

```python
#Lets look at a quick snapshot of customer buying habits
df2_items = df2.groupby('FullName')['Items.qty', 'Items.price', 'Items.cost', 'profit'].sum()
df2_items.sample(20)
```

<details>
  <summary>Customer buying habits</summary>

  <div>
  <style scoped>
      .dataframe tbody tr th:only-of-type {
          vertical-align: middle;
      }

      .dataframe tbody tr th {
          vertical-align: top;
      }

      .dataframe thead th {
          text-align: right;
      }
  </style>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>Items.qty</th>
        <th>Items.price</th>
        <th>Items.cost</th>
        <th>profit</th>
      </tr>
      <tr>
        <th>FullName</th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>DavidCislo</th>
        <td>1.0</td>
        <td>26.09</td>
        <td>17.99</td>
        <td>8.100</td>
      </tr>
      <tr>
        <th>StevenAlexander</th>
        <td>1.0</td>
        <td>39.35</td>
        <td>28.11</td>
        <td>11.240</td>
      </tr>
      <tr>
        <th>DarrylMiller</th>
        <td>1.0</td>
        <td>69.17</td>
        <td>53.21</td>
        <td>15.960</td>
      </tr>
      <tr>
        <th>olgapasko</th>
        <td>2.0</td>
        <td>9.99</td>
        <td>6.36</td>
        <td>1.815</td>
      </tr>
      <tr>
        <th>GarveyStenersen</th>
        <td>1.0</td>
        <td>160.04</td>
        <td>128.03</td>
        <td>32.010</td>
      </tr>
      <tr>
        <th>CherylMorgan</th>
        <td>1.0</td>
        <td>60.92</td>
        <td>46.50</td>
        <td>14.420</td>
      </tr>
      <tr>
        <th>sueminier</th>
        <td>2.0</td>
        <td>16.23</td>
        <td>6.49</td>
        <td>4.870</td>
      </tr>
      <tr>
        <th>RobertPurnick</th>
        <td>1.0</td>
        <td>114.99</td>
        <td>90.99</td>
        <td>24.000</td>
      </tr>
      <tr>
        <th>KimFreudenberger</th>
        <td>1.0</td>
        <td>25.00</td>
        <td>17.24</td>
        <td>7.760</td>
      </tr>
      <tr>
        <th>HarlanJuster</th>
        <td>3.0</td>
        <td>47.99</td>
        <td>44.00</td>
        <td>1.330</td>
      </tr>
      <tr>
        <th>donnahoey</th>
        <td>1.0</td>
        <td>151.94</td>
        <td>121.55</td>
        <td>30.390</td>
      </tr>
      <tr>
        <th>DanielFoster</th>
        <td>1.0</td>
        <td>57.19</td>
        <td>43.99</td>
        <td>13.200</td>
      </tr>
      <tr>
        <th>AntlanticSurgicenter</th>
        <td>1.0</td>
        <td>85.67</td>
        <td>66.93</td>
        <td>18.740</td>
      </tr>
      <tr>
        <th>VendelCsaszar</th>
        <td>1.0</td>
        <td>35.49</td>
        <td>35.49</td>
        <td>0.000</td>
      </tr>
      <tr>
        <th>MarkForte</th>
        <td>2.0</td>
        <td>36.82</td>
        <td>21.04</td>
        <td>15.780</td>
      </tr>
      <tr>
        <th>Mary Allbright</th>
        <td>1.0</td>
        <td>116.99</td>
        <td>103.00</td>
        <td>13.990</td>
      </tr>
      <tr>
        <th>SchivonneBishop</th>
        <td>1.0</td>
        <td>46.37</td>
        <td>33.60</td>
        <td>12.770</td>
      </tr>
      <tr>
        <th>BarryOxford</th>
        <td>1.0</td>
        <td>34.27</td>
        <td>24.48</td>
        <td>9.790</td>
      </tr>
      <tr>
        <th>TanyaGlover</th>
        <td>2.0</td>
        <td>12.78</td>
        <td>7.30</td>
        <td>2.740</td>
      </tr>
      <tr>
        <th>DavidKim</th>
        <td>1.0</td>
        <td>58.36</td>
        <td>44.55</td>
        <td>13.810</td>
      </tr>
    </tbody>
  </table>
  </div>
</details>


And we can take a look at the shipping costs in case that's something we want to analyze later on.

```python
df2_order = df2.groupby('FullName')['subtotal', 'shipping', 'total'].sum()
df2_order.sample(20)
```

<details>
  <summary>Shipping Costs</summary>

  <div>
  <style scoped>
      .dataframe tbody tr th:only-of-type {
          vertical-align: middle;
      }

      .dataframe tbody tr th {
          vertical-align: top;
      }

      .dataframe thead th {
          text-align: right;
      }
  </style>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>subtotal</th>
        <th>shipping</th>
        <th>total</th>
      </tr>
      <tr>
        <th>FullName</th>
        <th></th>
        <th></th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>SteveGonzalez</th>
        <td>19.45</td>
        <td>9.95</td>
        <td>29.40</td>
      </tr>
      <tr>
        <th>HelenBlair</th>
        <td>32.63</td>
        <td>9.95</td>
        <td>42.58</td>
      </tr>
      <tr>
        <th>adamhenninger</th>
        <td>4.88</td>
        <td>9.95</td>
        <td>15.26</td>
      </tr>
      <tr>
        <th>MarkDowling</th>
        <td>121.36</td>
        <td>9.95</td>
        <td>119.17</td>
      </tr>
      <tr>
        <th>SusanHixson</th>
        <td>13.99</td>
        <td>0.00</td>
        <td>13.99</td>
      </tr>
      <tr>
        <th>RonHutchison</th>
        <td>37.99</td>
        <td>6.73</td>
        <td>37.99</td>
      </tr>
      <tr>
        <th>AleksandraBaeva</th>
        <td>270.66</td>
        <td>0.00</td>
        <td>270.66</td>
      </tr>
      <tr>
        <th>DEBBIETRENT</th>
        <td>171.91</td>
        <td>9.95</td>
        <td>164.67</td>
      </tr>
      <tr>
        <th>douglasperry</th>
        <td>112.87</td>
        <td>7.17</td>
        <td>101.58</td>
      </tr>
      <tr>
        <th>renee Pallotta</th>
        <td>24.40</td>
        <td>9.95</td>
        <td>35.30</td>
      </tr>
      <tr>
        <th>PurleyNewson</th>
        <td>20.77</td>
        <td>9.95</td>
        <td>30.72</td>
      </tr>
      <tr>
        <th>Jannett Stewart</th>
        <td>68.89</td>
        <td>0.00</td>
        <td>68.89</td>
      </tr>
      <tr>
        <th>ChristaDedebant</th>
        <td>747.23</td>
        <td>29.85</td>
        <td>777.08</td>
      </tr>
      <tr>
        <th>DianaPaulsen</th>
        <td>63.70</td>
        <td>9.95</td>
        <td>73.65</td>
      </tr>
      <tr>
        <th>T. JohnThomas</th>
        <td>23.27</td>
        <td>9.95</td>
        <td>33.22</td>
      </tr>
      <tr>
        <th>Donald Ellis</th>
        <td>57.19</td>
        <td>0.00</td>
        <td>56.12</td>
      </tr>
      <tr>
        <th>PatriciaSleeper</th>
        <td>14.85</td>
        <td>0.00</td>
        <td>14.85</td>
      </tr>
      <tr>
        <th>RHONDABRYAN</th>
        <td>13.13</td>
        <td>9.95</td>
        <td>21.77</td>
      </tr>
      <tr>
        <th>JoseAlva</th>
        <td>66.94</td>
        <td>9.95</td>
        <td>76.89</td>
      </tr>
      <tr>
        <th>HimanshuMisra</th>
        <td>24.99</td>
        <td>0.00</td>
        <td>24.99</td>
      </tr>
    </tbody>
  </table>
  </div>
</details>

Now, we'll create the 'recency' aspect of our Recency-Frequency-Monetary table for later.

```python
df3 = df2.copy()
df3.dropna(inplace=True)
df3['placed_date'] = df3['placed_date'].transform(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))

#Newest orders
df_order_new = df3.groupby('id_x')['placed_date'].max()

#Oldest orders
df_order_old = df3.groupby('id_x')['placed_date'].min()

df_order_old = pd.DataFrame(df_order_old)
df_order_new = pd.DataFrame(df_order_new)
df_order = pd.DataFrame()

df_order[['first_order', 'recent_order']] = pd.merge(df_order_old, df_order_new, left_on = 'id_x', right_on= 'id_x')
```

Now that we've done a good bit of feature engineering and analysis, it's time to do the actual modeling and segmentation.

# Modeling

The first thing that we'll use is PCA, which helps to reduce the amount of features so that we can cluster. PCA can make it difficult to determine WHY the customers are segmented the way they are, so we will also be segmenting based on profit and recency, then comparing the 2 methods.

## PCA

```python
#PCA only works on numerical datatypes, so we'll select those out
numerical_features = df2.select_dtypes(exclude='object').columns

pca_df = df2[numerical_features]

pca_df.dropna(inplace=True)

#Create the Principal Components
pca_2 = PCA(n_components=2).fit(pca_df)

#Most of the variance in our features can be explained with 1 PC
#We will include 2 for ease of visualization
print(pca_2.explained_variance_ratio_)
```

![Graph1](/assets/Project4/Project4Graph1.png)

Now, we'll create a DataFrame of the principal components and add it to our df2 for later use.

```python
#Create a PC DataFrame
principalcomponents = pca_2.fit_transform(pca_df)

principal_df = pd.DataFrame(data = principalcomponents, columns = ['principal component 1', 'principal component 2'])

df2.dropna(inplace = True)
#Make sure theyre the same shape so they merge properly
principal_df.shape, df2.shape
```
((3569, 2), (3569, 31))

```python
df2 = pd.concat([df2, principal_df], axis = 1)
```

# Clustering with KMeans

Since clustering requires low dimensionality data, we could use PCA to cluster. As stated before, PCA can make it difficult to know WHY the clusters exist how they do, so we will create a table to cluster based on profit and recency of the previous purchase. Other variables we could use to cluster include: shipping type, which vendor the customer used, and the types of products (if we used NLP to group product types).

## Clustering based on Profit

```python
df_cluster = df2.groupby('id_x')['profit', 'placed_date'].agg({'profit': 'sum','placed_date': 'max'})
df_cluster.head()

df_cluster.dropna(inplace = True)

X = pd.DataFrame(df_cluster)

#Start off with a large amount of clusters to analyze the optemization
K = KMeans(n_clusters = 9).fit(X)
```

With our cluster fit, it's time to determine the optimal amount of groups for our customers. We will use yellowbrick's clustering visualizers to help out with this. Silhouette scores closer to 1 are better (ranked from 0-1).

![Graph2](/assets/Project4/Project4Graph2.png)

![Graph3](/assets/Project4/Project4Graph3.png)
![Graph4](/assets/Project4/Project4Graph4.png)
![Graph5](/assets/Project4/Project4Graph5.png)

According to the graphs, 4 or 5 clusters would be best. We'll choose 4 for now.

```python
K = KMeans(n_clusters = 4).fit(X)
silhouette_score(X, K.labels_)
```
0.6309803241979693

```python
#Now we'll add the cluster labels to our final DataFrame that we'll use for reporting
df_cluster['cluster'] = K.labels_
df_cluster.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>profit</th>
      <th>placed_date</th>
      <th>cluster</th>
    </tr>
    <tr>
      <th>id_x</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3.0</th>
      <td>17.0625</td>
      <td>1.449604e+09</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>6.9400</td>
      <td>1.386780e+09</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>29.6800</td>
      <td>1.441905e+09</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>11.2000</td>
      <td>1.388156e+09</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>13.7900</td>
      <td>1.389303e+09</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

## Clustering based on PCA

We will follow the same process to determine cluster labels using PCA.

```python
#Start off with a large amount of clusters to analyze the optemization
K = KMeans(n_clusters = 7).fit(principal_df)
```

![Graph6](/assets/Project4/Project4Graph6.png)
![Graph7](/assets/Project4/Project4Graph7.png)
![Graph8](/assets/Project4/Project4Graph8.png)
![Graph9](/assets/Project4/Project4Graph9.png)

According to the graphs, 3 clusters is best for the PCA model.

```python
K = KMeans(n_clusters = 3).fit(X)
silhouette_score(X, K.labels_)
```

0.6402889003646225

We can see that the silhouette score for the PCA model is SLIGHTLY better than the previous model, but the PCA clusters may not be able to give us actionable information.

# RFM Table

Our final task, create the RFM table to report to our boss. We will include the clusters as well so they know who to reach out to with certain promotions.

```python
df_cluster = pd.merge(df_cluster, df_order, left_on= 'id_x', right_on= 'id_x')
df_cluster.head(30)
```

<details>
  <summary>RFM Table</summary>

  <div>
  <style scoped>
      .dataframe tbody tr th:only-of-type {
          vertical-align: middle;
      }

      .dataframe tbody tr th {
          vertical-align: top;
      }

      .dataframe thead th {
          text-align: right;
      }
  </style>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>profit</th>
        <th>placed_date</th>
        <th>cluster</th>
        <th>first_order</th>
        <th>recent_order</th>
      </tr>
      <tr>
        <th>id_x</th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>3.0</th>
        <td>17.062500</td>
        <td>1.449604e+09</td>
        <td>0</td>
        <td>2013-12-03 09:07:35</td>
        <td>2015-12-08 11:40:52</td>
      </tr>
      <tr>
        <th>4.0</th>
        <td>6.940000</td>
        <td>1.386780e+09</td>
        <td>2</td>
        <td>2013-12-11 08:44:23</td>
        <td>2013-12-11 08:44:23</td>
      </tr>
      <tr>
        <th>5.0</th>
        <td>29.680000</td>
        <td>1.441905e+09</td>
        <td>3</td>
        <td>2013-12-12 07:19:59</td>
        <td>2015-09-10 10:07:59</td>
      </tr>
      <tr>
        <th>7.0</th>
        <td>11.200000</td>
        <td>1.388156e+09</td>
        <td>2</td>
        <td>2013-12-27 06:52:27</td>
        <td>2013-12-27 06:52:27</td>
      </tr>
      <tr>
        <th>8.0</th>
        <td>13.790000</td>
        <td>1.389303e+09</td>
        <td>2</td>
        <td>2014-01-09 13:33:36</td>
        <td>2014-01-09 13:33:36</td>
      </tr>
      <tr>
        <th>10.0</th>
        <td>0.000000</td>
        <td>1.390510e+09</td>
        <td>2</td>
        <td>2014-01-23 12:38:36</td>
        <td>2014-01-23 12:38:36</td>
      </tr>
      <tr>
        <th>11.0</th>
        <td>0.000000</td>
        <td>1.390573e+09</td>
        <td>2</td>
        <td>2014-01-24 06:19:15</td>
        <td>2014-01-24 06:19:15</td>
      </tr>
      <tr>
        <th>12.0</th>
        <td>3.970000</td>
        <td>1.390613e+09</td>
        <td>2</td>
        <td>2014-01-24 17:30:19</td>
        <td>2014-01-24 17:30:19</td>
      </tr>
      <tr>
        <th>13.0</th>
        <td>17.260000</td>
        <td>1.424206e+09</td>
        <td>1</td>
        <td>2014-01-27 12:54:01</td>
        <td>2015-02-17 12:46:29</td>
      </tr>
      <tr>
        <th>14.0</th>
        <td>0.000000</td>
        <td>1.391012e+09</td>
        <td>2</td>
        <td>2014-01-29 08:13:37</td>
        <td>2014-01-29 08:13:37</td>
      </tr>
      <tr>
        <th>15.0</th>
        <td>7.620000</td>
        <td>1.391363e+09</td>
        <td>2</td>
        <td>2014-02-02 09:43:24</td>
        <td>2014-02-02 09:43:24</td>
      </tr>
      <tr>
        <th>21.0</th>
        <td>0.000000</td>
        <td>1.391619e+09</td>
        <td>2</td>
        <td>2014-02-05 08:45:32</td>
        <td>2014-02-05 08:45:32</td>
      </tr>
      <tr>
        <th>22.0</th>
        <td>2.565000</td>
        <td>1.391815e+09</td>
        <td>2</td>
        <td>2014-02-07 15:17:52</td>
        <td>2014-02-07 15:17:52</td>
      </tr>
      <tr>
        <th>23.0</th>
        <td>19.850000</td>
        <td>1.420601e+09</td>
        <td>1</td>
        <td>2014-02-08 14:01:19</td>
        <td>2015-01-06 19:20:00</td>
      </tr>
      <tr>
        <th>24.0</th>
        <td>9.120000</td>
        <td>1.392048e+09</td>
        <td>2</td>
        <td>2014-02-10 08:07:54</td>
        <td>2014-02-10 08:07:54</td>
      </tr>
      <tr>
        <th>25.0</th>
        <td>0.000000</td>
        <td>1.392049e+09</td>
        <td>2</td>
        <td>2014-02-10 08:10:47</td>
        <td>2014-02-10 08:10:47</td>
      </tr>
      <tr>
        <th>26.0</th>
        <td>0.000000</td>
        <td>1.392054e+09</td>
        <td>2</td>
        <td>2014-02-10 09:38:57</td>
        <td>2014-02-10 09:38:57</td>
      </tr>
      <tr>
        <th>27.0</th>
        <td>93.870000</td>
        <td>1.404227e+09</td>
        <td>2</td>
        <td>2014-02-11 07:46:40</td>
        <td>2014-07-01 07:57:15</td>
      </tr>
      <tr>
        <th>30.0</th>
        <td>0.000000</td>
        <td>1.392312e+09</td>
        <td>2</td>
        <td>2014-02-13 09:17:08</td>
        <td>2014-02-13 09:17:08</td>
      </tr>
      <tr>
        <th>31.0</th>
        <td>4.366667</td>
        <td>1.392393e+09</td>
        <td>2</td>
        <td>2014-02-14 07:43:48</td>
        <td>2014-02-14 07:43:48</td>
      </tr>
      <tr>
        <th>32.0</th>
        <td>10.310000</td>
        <td>1.392394e+09</td>
        <td>2</td>
        <td>2014-02-14 08:08:30</td>
        <td>2014-02-14 08:08:30</td>
      </tr>
      <tr>
        <th>33.0</th>
        <td>3.660000</td>
        <td>1.392485e+09</td>
        <td>2</td>
        <td>2014-02-15 09:24:01</td>
        <td>2014-02-15 09:24:01</td>
      </tr>
      <tr>
        <th>35.0</th>
        <td>6.110000</td>
        <td>1.393012e+09</td>
        <td>2</td>
        <td>2014-02-21 11:49:50</td>
        <td>2014-02-21 11:49:50</td>
      </tr>
      <tr>
        <th>37.0</th>
        <td>21.910000</td>
        <td>1.393253e+09</td>
        <td>2</td>
        <td>2014-02-24 06:40:55</td>
        <td>2014-02-24 06:40:55</td>
      </tr>
      <tr>
        <th>39.0</th>
        <td>6.740000</td>
        <td>1.393370e+09</td>
        <td>2</td>
        <td>2014-02-25 15:13:24</td>
        <td>2014-02-25 15:13:24</td>
      </tr>
      <tr>
        <th>40.0</th>
        <td>8.925000</td>
        <td>1.393524e+09</td>
        <td>2</td>
        <td>2014-02-27 10:01:14</td>
        <td>2014-02-27 10:01:14</td>
      </tr>
      <tr>
        <th>41.0</th>
        <td>20.400000</td>
        <td>1.393537e+09</td>
        <td>2</td>
        <td>2014-02-27 13:40:19</td>
        <td>2014-02-27 13:40:19</td>
      </tr>
      <tr>
        <th>42.0</th>
        <td>0.720000</td>
        <td>1.393681e+09</td>
        <td>2</td>
        <td>2014-03-01 05:37:14</td>
        <td>2014-03-01 05:37:14</td>
      </tr>
      <tr>
        <th>43.0</th>
        <td>11.200000</td>
        <td>1.393712e+09</td>
        <td>2</td>
        <td>2014-03-01 14:12:25</td>
        <td>2014-03-01 14:12:25</td>
      </tr>
      <tr>
        <th>44.0</th>
        <td>9.310000</td>
        <td>1.393862e+09</td>
        <td>2</td>
        <td>2014-03-03 08:01:37</td>
        <td>2014-03-03 08:01:37</td>
      </tr>
    </tbody>
  </table>
  </div>
</details>

Here we have a table grouped by the customer ID, that shows us the profit we have earned from each customer, the time of their last order, and their cluster number. The clusters can be analyzed further to determine which type of product/offer will be pushed to them.

That's all for now! As you can tell, there is a lot more that can be done with this data, but due to time and objective constraints I have limited this project to what is done here. If the boss wants us to dive a bit deeper using NLP on our product descriptions, or analyze the clusters more/differently that can be done in further iterations of the project.

Thank you for taking the time to read through this project. If you have any questions, please feel free to reach out to me via email!
