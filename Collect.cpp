#include "Collect.h"

vector<int> FaceIDQuery(string query)
{
    SQLiteDBConnection conn;
    conn.open(DB_FILE);

	int ID;
    vector<int> faceIds;

    SQLiteStmt *filterQueryStmt = conn.prepare(string(query));

	if (filterQueryStmt == 0)
	{
		cout << "Error preparing !" << endl;
		return faceIds;
	}

    bool allOK = true;
	int res = 0;

	do
	{
		res = conn.step(filterQueryStmt);

		if (res == SQLITE_ROW)
		{
			allOK = allOK && filterQueryStmt->readIntColumn(0, ID);

			if (allOK)
				faceIds.push_back(ID);
			else
				break;
		}
	} while (res == SQLITE_ROW);

    conn.close();

    return faceIds;
}

FaceData* getFaceData(int id, SQLiteDBConnection* conn)
{
    FaceDataByIDsQuery faceDataSqlQuery;

    vector<int> vid;
    vid.push_back(id);

	faceDataSqlQuery.queryIds = vid;
	faceDataSqlQuery.exec(conn);

	std::map<int, FaceData*> annotations = faceDataSqlQuery.data;

	FaceData *currFaceAnnotation = annotations[id];
    //FaceData has to be copied as it is destroyed on destruction of the query//
    FaceData *cpyCurrFaceAnno = new FaceData(*currFaceAnnotation);

	return cpyCurrFaceAnno;
}

vector<int> kpfilter(vector<int> alldata)
{
    SQLiteDBConnection conn;
    conn.open(DB_FILE);

    vector<int> seldata;

    for (int i = 0; i < alldata.size(); i++)
    {
        FaceData *face = getFaceData(alldata[i], &conn);
        face->loadFeatureCoords(&conn);

        bool exfts[22];
        for (int j = 0; j < 22; j++) exfts[j] = false;

        vector<int> fetids = face->getFeaturesCoords()->getFeatureIds();

        for (int j = 0; j < fetids.size(); j++)
            exfts[fetids[j]] = true;

        double yaw = face->getPose()->yaw;

        if ((yaw >= 2*CV_PI/6.0f) && (yaw <= 3*CV_PI/6.0f))
        {
            int inds[11] = {4, 5, 6, 10, 11, 12, 15, 16, 19, 20, 21};
            bool chk = true;

            for (int j = 0; j < 11; j++)
                if (!exfts[inds[j]]) chk = false;

            if (chk)
                seldata.push_back(alldata[i]);
        }
        else if ((yaw >= 1*CV_PI/6.0f) && (yaw <= 2*CV_PI/6.0f))
        {
            int inds[11] = {4, 5, 6, 10, 11, 12, 15, 16, 19, 20, 21};
            bool chk = true;

            for (int j = 0; j < 11; j++)
                if (!exfts[inds[j]]) chk = false;

            if (!exfts[1] && !exfts[2] && !exfts[3]) chk = false;
            if (!exfts[7] && !exfts[8] && !exfts[9]) chk = false;
            if (!exfts[18]) chk = false;

            if (chk)
                seldata.push_back(alldata[i]);
        }
        else if ((yaw >= -1*CV_PI/6.0f) && (yaw <= 1*CV_PI/6.0f))
        {
            int inds[19] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21};
            bool chk = true;

            for (int j = 0; j < 19; j++)
                if (!exfts[inds[j]]) chk = false;

            if (chk)
                seldata.push_back(alldata[i]);
        }
        else if ((yaw >= -2*CV_PI/6.0f) && (yaw <= -1*CV_PI/6.0f))
        {
            int inds[11] = {1, 2, 3, 7, 8, 9, 14, 15, 18, 19, 21};
            bool chk = true;

            for (int j = 0; j < 11; j++)
                if (!exfts[inds[j]]) chk = false;

            if (!exfts[4] && !exfts[5] && !exfts[6]) chk = false;
            if (!exfts[10] && !exfts[11] && !exfts[12]) chk = false;
            if (!exfts[20]) chk = false;

            if (chk)
                seldata.push_back(alldata[i]);
        }
        else if ((yaw >= -3*CV_PI/6.0f) && (yaw <= -2*CV_PI/6.0f))
        {
            int inds[11] = {1, 2, 3, 7, 8, 9, 14, 15, 18, 19, 21};
            bool chk = true;

            for (int j = 0; j < 11; j++)
                if (!exfts[inds[j]]) chk = false;

            if (chk)
                seldata.push_back(alldata[i]);
        }

        delete face;
    }

    conn.close();

    return seldata;
}

void collectTrainingEllipses(char* fname)
{
    int nranges = 8;
    int nrang2 = nranges/2;

    vector<vector<int> > faceinds;
    faceinds.resize(nranges);

    stringstream qstr;

    for (int i = 0; i < nranges; i++)
    {
        qstr.str(""); qstr.clear();
        qstr << "SELECT faces.face_id FROM faces,faceellipse WHERE faces.face_id = faceellipse.face_id AND ";
        qstr << "theta >= " << (i - nrang2)*CV_PI/nranges << " AND " << " theta <= " << (i - nrang2 + 1)*CV_PI/nranges
        << " ORDER BY faces.face_id";
        faceinds[i] = FaceIDQuery(qstr.str());

        if ((i > 1) && (i < 6))
            faceinds[i].resize((faceinds[i].size()/4)*3);
        else
            faceinds[i].resize(350 < faceinds[i].size()?350:faceinds[i].size());

        cout << faceinds[i].size() << endl;
    }

    for (int i = 0; i < nranges; i++)
        sort(faceinds[i].begin(), faceinds[i].end());

    vector<int> faceunion;
    vector<int> faceellipse = faceinds[0];

    for (int i = 1; i < nranges; i++)
    {
        faceunion = faceellipse;
        faceellipse.clear();
        set_union(faceunion.begin(), faceunion.end(), faceinds[i].begin(), faceinds[i].end(), back_inserter(faceellipse));
    }

    cout << "all: " << faceellipse.size() << endl;

    ofstream fs(fname);

    fs << faceellipse.size() << endl;

    for (int i = 0; i < faceellipse.size(); i++)
        fs << faceellipse[i] << endl;

    fs.close();
}

void collectTrainingPoses(char* fname)
{
    int nranges = 8;
    int nrang2 = nranges/2;

    vector<vector<int> > faceinds;
    faceinds.resize(nranges);

    stringstream qstr;

    for (int i = 0; i < nranges; i++)
    {
        qstr.str(""); qstr.clear();
        qstr << "SELECT faces.face_id FROM faces,facepose WHERE faces.face_id = facepose.face_id AND ";
        qstr << "yaw >= " << (i - nrang2)*CV_PI/nranges << " AND " << " yaw <= " << (i - nrang2 + 1)*CV_PI/nranges
        << " ORDER BY faces.face_id";
        faceinds[i] = FaceIDQuery(qstr.str());

        faceinds[i] = kpfilter(faceinds[i]);
        faceinds[i].resize(200 < faceinds[i].size()?200:faceinds[i].size());

        cout << faceinds[i].size() << endl;
    }

    cout << endl;

    for (int i = 0; i < nranges; i++)
        sort(faceinds[i].begin(), faceinds[i].end());

    vector<int> faceunion;
    vector<int> faceyaw = faceinds[0];

    for (int i = 1; i < nranges; i++)
    {
        faceunion = faceyaw;
        faceyaw.clear();
        set_union(faceunion.begin(), faceunion.end(), faceinds[i].begin(), faceinds[i].end(), back_inserter(faceyaw));
    }

    for (int i = 0; i < nranges; i++)
    {
        qstr.str(""); qstr.clear();
        qstr << "SELECT faces.face_id FROM faces,facepose WHERE faces.face_id = facepose.face_id AND ";
        qstr << "pitch >= " << (i - nrang2)*CV_PI/nranges << " AND " << " pitch <= " << (i - nrang2 + 1)*CV_PI/nranges
        << " ORDER BY faces.face_id";
        faceinds[i] = FaceIDQuery(qstr.str());

        faceinds[i] = kpfilter(faceinds[i]);

        if ((i > 1) && (i < 5))
            faceinds[i].resize(300 < faceinds[i].size()?300:faceinds[i].size());

        cout << faceinds[i].size() << endl;
    }

    for (int i = 0; i < nranges; i++)
        sort(faceinds[i].begin(), faceinds[i].end());

    vector<int> facepitch = faceinds[0];

    for (int i = 1; i < nranges; i++)
    {
        faceunion = facepitch;
        facepitch.clear();
        set_union(faceunion.begin(), faceunion.end(), faceinds[i].begin(), faceinds[i].end(), back_inserter(facepitch));
    }

    vector<int> faceall;

    set_union(faceyaw.begin(), faceyaw.end(), facepitch.begin(), facepitch.end(), back_inserter(faceall));

    cout << "all: " << faceall.size() << endl;

    ofstream fs(fname);

    fs << faceall.size() << endl;

    for (int i = 0; i < faceall.size(); i++)
        fs << faceall[i] << endl;

    fs.close();
}

void collectTestEllipses(char* fname, char* trainingfile)
{
    ifstream ifs(trainingfile);

    int sz;
    ifs >> sz;

    vector<int> trinds(sz);

    for (int i = 0; i < sz; i++)
        ifs >> trinds[i];

    ifs.close();

    stringstream qstr;

    qstr.str(""); qstr.clear();
    qstr << "SELECT faces.face_id FROM faces,faceellipse WHERE faces.face_id = faceellipse.face_id AND ";
    qstr << "theta >= " << -CV_PI/2.0f << " AND " << " theta <= " << CV_PI/2.0f
    << " ORDER BY faces.face_id";
    vector<int> allfaces = FaceIDQuery(qstr.str());

    sort(trinds.begin(), trinds.end());
    sort(allfaces.begin(), allfaces.end());

    vector<int> selfaces;
    set_difference(allfaces.begin(), allfaces.end(), trinds.begin(), trinds.end(), back_inserter(selfaces));

    CvMat* rdmat = randperm(selfaces.size());
    int numret = (2000 < selfaces.size()?2000:selfaces.size());

    vector<int> retfaces(numret);

    for (int i = 0; i < numret; i++)
    {
        int idx = CV_MAT_ELEM(*rdmat, int, 0, i);
        retfaces[i] = selfaces[idx];
    }

    cout << retfaces.size() << endl;

    ofstream fs(fname);

    fs << retfaces.size() << endl;

    for (int i = 0; i < retfaces.size(); i++)
        fs << retfaces[i] << endl;

    fs.close();
}

void collectTestPoses(char* fname, char* trainingfile)
{
    ifstream ifs(trainingfile);

    int sz;
    ifs >> sz;

    vector<int> trinds(sz);

    for (int i = 0; i < sz; i++)
        ifs >> trinds[i];

    ifs.close();

    stringstream qstr;

    qstr.str(""); qstr.clear();
    qstr << "SELECT faces.face_id FROM faces,facepose WHERE faces.face_id = facepose.face_id AND ";
    qstr << "yaw >= " << -CV_PI/2.0f << " AND " << " yaw <= " << CV_PI/2.0f << " AND ";
    qstr << "pitch >= " << -CV_PI/2.0f << " AND " << " pitch <= " << CV_PI/2.0f
    << " ORDER BY faces.face_id";
    vector<int> allfaces = FaceIDQuery(qstr.str());

    sort(trinds.begin(), trinds.end());
    sort(allfaces.begin(), allfaces.end());

    vector<int> selfaces;
    set_difference(allfaces.begin(), allfaces.end(), trinds.begin(), trinds.end(), back_inserter(selfaces));

    CvMat* rdmat = randperm(selfaces.size());
    int numret = (2000 < selfaces.size()?2000:selfaces.size());

    vector<int> retfaces(numret);

    for (int i = 0; i < numret; i++)
    {
        int idx = CV_MAT_ELEM(*rdmat, int, 0, i);
        retfaces[i] = selfaces[idx];
    }

    cout << retfaces.size() << endl;

    ofstream fs(fname);

    fs << retfaces.size() << endl;

    for (int i = 0; i < retfaces.size(); i++)
        fs << retfaces[i] << endl;

    fs.close();
}

