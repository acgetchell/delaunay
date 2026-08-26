#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

justfile_path="${1:-justfile}"
declaration_re='^[[:space:]]*just_version[[:space:]]*:=[[:space:]]*(.*)$'
versions=()

while IFS= read -r line; do
	if [[ "$line" =~ $declaration_re ]]; then
		value="${BASH_REMATCH[1]}"
		value="${value%%#*}"
		value="${value##+([[:space:]])}"
		value="${value%%+([[:space:]])}"

		if [[ ${#value} -ge 2 ]]; then
			first="${value:0:1}"
			last="${value: -1}"
			if [[ "$first" == "$last" && ("$first" == '"' || "$first" == "'") ]]; then
				value="${value:1:${#value}-2}"
			fi
		fi
		versions+=("$value")
	fi
done <"$justfile_path"

if [[ ${#versions[@]} -ne 1 || -z "${versions[0]}" ]]; then
	echo "Could not resolve exactly one non-empty just_version from $justfile_path" >&2
	exit 1
fi
if [[ ! "${versions[0]}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)?$ ]]; then
	echo "Resolved invalid just_version from $justfile_path: ${versions[0]}" >&2
	exit 1
fi

printf '%s\n' "${versions[0]}"
