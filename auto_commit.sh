#!/bin/bash
cd /home/lustre_p/ahmed.omar/workspace/exo_de_project || exit 1

# Stage everything (gitignore handles exclusions)
git add -A

# Only commit if there are staged changes
if ! git diff --cached --quiet; then
    git commit -m "auto: $(date '+%Y-%m-%d %H:%M')"
    git push origin main 2>&1 || echo "push failed at $(date)"
fi
