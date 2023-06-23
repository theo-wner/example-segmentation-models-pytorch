import os
import shutil

################################################################################
# Skript, das das Löschen von Ordnern möglich macht, die mit python os 
# erstellt wurden (Sonst keine Berechtigung)
################################################################################

# Datei löschen
#os.remove('./Abbildungen/pred_0.png')

# Leeres Verzeichnis löschen
#os.rmdir('./Abbildungen/')

# Verzeichnis und seine Inhalte löschen
shutil.rmtree('./Abbildungen/')