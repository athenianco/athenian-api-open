.PHONY: build-native
build-native:
	cmake -S athenian/api/sentry_native -B athenian/api/sentry_native/build -D SENTRY_BACKEND=crashpad -D SENTRY_BUILD_EXAMPLES=OFF -D SENTRY_BUILD_TESTS=OFF -D CMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build athenian/api/sentry_native/build --parallel
	cmake -S athenian/api/mimalloc -B athenian/api/mimalloc/build -D mi_cflags=-flto -D MI_BUILD_STATIC=OFF -D MI_BUILD_OBJECT=OFF -D MI_BUILD_TESTS=OFF -D MI_INSTALL_TOPLEVEL=ON -D MI_USE_CXX=OFF -D CMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build athenian/api/mimalloc/build --parallel

.PHONY: install-native
install-native: build-native
	sudo cmake --install athenian/api/sentry_native/build
	sudo cmake --install athenian/api/mimalloc/build

PHONY: install-native-user
install-native-user: build-native
	cmake --install athenian/api/sentry_native/build
	cmake --install athenian/api/mimalloc/build

PHONY: clean-native
clean-native:
	rm -rf athenian/api/sentry_native/build
	rm -rf athenian/api/mimalloc/build
