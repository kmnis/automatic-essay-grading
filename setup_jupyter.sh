pip install -r requirements.txt

jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user

jupyter nbextension enable --py --sys-prefix appmode && \
        jupyter serverextension enable --py --sys-prefix appmode && \
        jupyter nbextension enable highlight_selected_word/main --sys-prefix && \
        jupyter nbextension enable codefolding/main --sys-prefix && \
        jupyter nbextension enable execute_time/ExecuteTime --sys-prefix
