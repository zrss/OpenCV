#!/bin/bash

if [ ! -e Makefile ]; then
	cmake ../Code/
fi

make