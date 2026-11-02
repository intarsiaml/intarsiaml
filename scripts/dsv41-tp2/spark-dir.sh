# Sourced by the scripts in this directory: where a lane's source, binaries and logs live on the Sparks.
#   SPARK_DIR=~/intarsia-mylane scripts/dsv41-tp2/build.sh       (default ~/intarsia-dsv41-graphs)
# Any spelling under the home directory works (~/x, $HOME/x, x), also a ~ the local shell already expanded.
# Afterwards SPARK_REL is the path relative to the remote home (for scp and rsync targets) and SPARK_DIR is the literal text
# `$HOME/<rel>`, for the remote shell to expand. Rank caches and fixtures stay in ~/intarsia-dsv41-tp2: they are read only.
_d=${SPARK_DIR:-intarsia-dsv41-graphs}
_d=${_d#"$HOME"/}
_d=${_d#\~/}
_d=${_d#\$HOME/}
_d=${_d%/}
case "/$_d/" in
  //|/./|/../|*/../*|*/./*|*//*|*" "*) echo "SPARK_DIR '${SPARK_DIR:-}': needs a directory under the home directory" >&2; exit 2 ;;
esac
[[ $_d == /* ]] && { echo "SPARK_DIR '${SPARK_DIR:-}': needs a directory under the home directory" >&2; exit 2; }
SPARK_REL=$_d
SPARK_DIR="\$HOME/$_d"
unset _d
