import json

OpenEarthMap_classes = ['background','bareland,barren','grass','pavement','road','tree,forest','water,river','cropland','building,roof,house']

potsdam_classes = ['road,parking lot','building','low vegetation','tree','car','clutter,background']

LoveDA_classes = ['background', 'building,roof,house', 'road', 'water','barren' ,'forest','agricultural']

iSAID_classes = ['background','ship','store tank','baseball diamond','tennis court','basketball court','ground track field','bridge','large vehicle','small vehicle','helicopter','swimming pool','roundabout','soccer ball field','plane','harbor']

uavid_classes = ['background','building','road','car','tree','vegetation','human']

udd5_classes = ['vegetation','building','road','vehicle','background']

vaihingen_classes=['impervious surface','building','low vegetation','tree','car','clutter']

vdd_classes=['background','facade','road','vegetation','vehicle','roof','water']

road_classes = ['background','road']

water_classes = ['background','water']

building_classes = ['background','building']

if __name__ == '__main__':
    print("OpenEarthMap:", len(OpenEarthMap_classes))
    print("potsdam:", len(potsdam_classes))
    print("LoveDA:", len(LoveDA_classes))
    print("iSAID:", len(iSAID_classes))
    print("uavid:", len(uavid_classes))
    print("udd5:", len(udd5_classes))
    print("vaihingen:", len(vaihingen_classes))
    print("vdd:", len(vdd_classes))
    print("road:", len(road_classes))
    print("water:", len(water_classes))
    print("building:", len(building_classes))