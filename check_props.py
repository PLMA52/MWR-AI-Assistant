from neo4j import GraphDatabase
d = GraphDatabase.driver('neo4j+s://551c1b37.databases.neo4j.io', auth=('neo4j','RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g'))
s = d.session()
r = s.run('MATCH (z:ZipCode) WHERE z.pct_no_diploma IS NOT NULL RETURN keys(z) AS k LIMIT 1')
print(r.single()['k'])
d.close()
