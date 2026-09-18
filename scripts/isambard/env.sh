# Environment for the Isambard-AI experiment jobs: repository, cache root, uv locations, venv.
# Source on the login node before sbatch (jobs inherit it) and at the top of every job.
export AGT_REPO="$PROJECTDIR/$USER/AbstractGradientTraining"
export AGT_ROOT="$PROJECTDIR/$USER/agt"
export UV_CACHE_DIR="$SCRATCHDIR/uv-cache"   # keeps multi-GB wheels out of the 100 GiB $HOME
export UV_LINK_MODE=copy                      # cache and venv are on different filesystems
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
if [ -f "$AGT_REPO/.venv/bin/activate" ]; then
    source "$AGT_REPO/.venv/bin/activate"
fi
