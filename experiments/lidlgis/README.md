Are Lidl supermarkets usually located in a less expensive to live area compared to more 'premium' supermarkets like Albert Heijn?
This is a question that came up the other day, and it just so happens to be a great excuse for playing with geospatial data.

## Getting the data
We need three datasets:
1. A metric for cost of living in particular areas. 
For this I will use the sales price per square meter for houses in Amsterdam, available at https://maps.amsterdam.nl/open_geodata/.
Unfortunately I could not find a map for the entire country, but one large city is hopefully good enough.
Note that we could have also used rental prices instead (there is probably a dataset for that).

2. Locations of Lidl branches. 
I used the branch finder tool at https://www.lidl.nl. 
If you search for Amsterdam with the Network inspector open, you can extract the JSON data in `lidl_amsterdam.json`.

3. Locations of Albert Heijn branches.
We use the same trick here using the branch finder at https://www.ah.nl/winkels/amsterdam.
The response JSON is stored in `ah_amsterdam.json`.