'''
This is a script to read in all the RCP and geoengineering pp data from HadGEM2-ES, calculate aragonite saturation states, and save output as netcdf3 files
'''

import numpy as np
import iris
import carbchem
import iris.analysis
import iris.analysis.cartography
import time

#time.sleep(60*60*10)

c_stash_101=iris.AttributeConstraint(STASH='m02s00i101')
c_stash_102=iris.AttributeConstraint(STASH='m02s00i102')
c_stash_103=iris.AttributeConstraint(STASH='m02s00i103')
c_stash_104=iris.AttributeConstraint(STASH='m02s00i104')

dir='/data/local/hador/mass_retrievals/rcp_geo_arag/'

g3=['aktwk','aktwl','aktwm']
g4=['aktwh','aktwi','aktwj']
rcp26=['ajnjm', 'kaadc', 'kaaec', 'kaafc']
rcp45=['ajnjg', 'kaadd', 'kaaed', 'kaafd']
rcp60=['ajnjs', 'kaade', 'kaaee', 'kaafe']
rcp85=['ajnji', 'kaadf', 'kaaef', 'kaaff']
co2_stab=['alytb']
geo_e=['amreb','amrec']

#runs=[rcp26,rcp45,rcp60,rcp85,co2_stab,geo_e]
runs=[g3,g4]

for i,dummy in enumerate(runs):
    size=np.size(dummy)
    for j in range(size):
        print 'processing '+runs[i][j]

        cube=iris.load(dir+runs[i][j]+'/*.pp')

        s101 = cube.extract(c_stash_101)
        s101=s101[0]
        s102 = cube.extract(c_stash_102)
        s102=s102[0]
        s103 = cube.extract(c_stash_103)
        s103=s103[0]
        s104 = cube.extract(c_stash_104)
        s104=s104[0]

        arag_ss_data = carbchem.carbchem(9,s101.data.fill_value,s101.data,s102.data*1000.0+35.0,s103.data/(1026.0*1000.0),s104.data/(1026.0*1000.0))

        arag_ss=s101.copy(data=arag_ss_data)
        arag_ss.long_name='aragnoite_saturation_state'

        iris.fileformats.netcdf.save(arag_ss, '/data/local/hador/arag_ss/'+runs[i][j]+'.nc', netcdf_format='NETCDF3_CLASSIC')

