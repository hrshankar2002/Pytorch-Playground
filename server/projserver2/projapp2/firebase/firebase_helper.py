def uploadimage(storage, DSIMG, ENDIMG):
    storage.child("dsfig1.png").put(DSIMG)
    storage.child("endfig1.png").put(ENDIMG)
    
    url_dsfig1 = storage.child("dsfig1.png").get_url(None)
    url_endfig1 = storage.child("endfig1.png").get_url(None)
    
    return url_dsfig1, url_endfig1