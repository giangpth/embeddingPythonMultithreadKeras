QT += core
QT -= gui

TARGET = MyMultiThread
CONFIG += console
CONFIG -= app_bundle

CONFIG += thread
CONFIG += c++11

TEMPLATE = app

SOURCES += main.cpp \
    test.cpp


INCLUDEPATH += /usr/include/python3.5m
LIBS += -L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu \
-lpython3.5m

HEADERS += \
    faceclassify.h \
    test.h
