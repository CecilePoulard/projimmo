#Fichier pour créer des raccourcis, i.e. regrouper des lignes sur le terminal

#@pip uninstall -y taxifare || :
# SYNTAXE pour l'appeler:
#make <some_action>
#

# SYNTAXE dans le fichier:
# nom_de_la_directive:
# <tab> directive avec différents arguments
# <tab> @ action "masqué" (i.e. l'action demandée ne sera pas affichée dans le terminal lors de l'execution)
# <tab> - (par défaut une action qui ne fonctionne pas dans le terminal, celles qui
# qui suivent ne seront pas executées. Sauf s'il y a un '-', alors elle sera executée quoi qu'il)
# <tab> -@

## Dans le terminal 'make install_package'
install_package:
	@pip uninstall -y projimmo || :
	@pip install -e .



##################### CLEANING #####################
#supprime tous les fichiers cachés
clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints
