#ifndef __COLLECT_H__
#define __COLLECT_H__

#include <iostream>
#include <fstream>
#include <vector>

#include "aflw/dbconn/SQLiteDBConnection.h"
#include "aflw/facedata/FaceData.h"
#include "aflw/querys/FaceDataByIDsQuery.h"

#include "debug.h"

#define DB_DIR "C:\\AFLW\\data"
#define DB_FILE "C:\\AFLW\\data\\aflw.sqlite"

void collectTrainingEllipses(char* fname);
void collectTrainingPoses(char* fname);
void collectTestEllipses(char* fname, char* trainingfile);
void collectTestPoses(char* fname, char* trainingfile);

#endif
