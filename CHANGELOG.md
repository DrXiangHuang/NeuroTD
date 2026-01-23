# 1.0.1
- Fixed `.gitignore` to include `CHANGELOG.md` and application subfolders.
- Improved command-line execution behavior for Matplotlib-based simulation scripts (previously developed primarily in VS Code interactive mode):
  - Added `plt.ion()` at the beginning of scripts to enable non-blocking interactive plotting during command-line execution.
  - Added `plt.show(block=True)` at the end of scripts to keep figures open when running outside interactive environments.
- Renamed signal.py to neuro_signal.py to avoid name conflicts with scipy.signal during script execution

# 1.0.0
- Initial release
