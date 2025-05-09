from astropy.table import Table


data = Table.read('./rdls.fits')
print(data)

print('fk5;')
for row in data:
    print(f'CIRCLE({row["RA"]}, {row["DEC"]}, 1000")')
