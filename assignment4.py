from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import csv
import logging
from functools import cmp_to_key
from typing import List, Tuple
import uuid



# Note (john): Make sure you use Python's logger to log
#              information about your program
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# friend_file = 'friends.txt'
# movie_file = 'movie_ratings.txt'

toyfriend = 'friends_testdata.txt'
toymovie = 'ratings_testdata.txt'

friend_res = 'friends_test2.txt'
movie_res = 'ratings_test2.txt'


# Generates unique operator IDs
def _generate_uuid():
    return uuid.uuid4()


# Custom tuple class with optional metadata
class ATuple:
    """Custom tuple.

    Attributes:
        tuple (Tuple): The actual tuple.
        metadata (string): The tuple metadata (e.g. provenance annotations).
        operator (Operator): A handle to the operator that produced the tuple.
    """
    def __init__(self, tuple, metadata=None, operator=None):
        self.tuple = tuple
        self.metadata = metadata
        self.operator = operator

    # Returns the lineage of self

    def lineage(self)-> List[ATuple]:
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        resultTup = self.operator.lineage(tuples=[self])
        return resultTup
        pass

    # Returns the Where-provenance of the attribute at index 'att_index' of self
    def where(self,att_index) -> List[Tuple]:
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        resultlist=[]
        resultTup = self.operator.where(att_index=att_index,tuples=[self])
        return resultTup
        pass

    # Returns the How-provenance of self
    def how(self) -> String:
        # YOUR CODE HERE (ONLY FOR TASK 3 IN ASSIGNMENT 2)
        if isinstance(self.operator,OrderBy):
            metalist = self.metadata
            metastring_avg=""
            for k in metalist:
                metastring_avg += k+","
            metastring_avg = "AVG( " + metastring_avg[0:-1] +" )"
            return metastring_avg
        return self.metadata
        pass

    # Returns the input tuples with responsibility \rho >= 0.5 (if any)
    def responsible_inputs(self) -> List[Tuple]:
        def getAvg(rlist,eleRemoval):
            total = 0;
            for i in rlist:
                total += i
            listlen = len(rlist)-eleRemoval
            if listlen == 0:
                return 0
            else:
                return round(total/listlen,2)

        intoOut = self.operator.input_to_output





        # resultTup = self.operator.lineage(self.tuple)

        # responList = []
        # for tuple in resultTup:
        #     tuple = (tuple.tuple,0.5)
        #     responList.append(tuple)

        # YOUR CODE HERE (ONLY FOR TASK 4 IN ASSIGNMENT 2)
        pass

    def __repr__(self):
        return str(self.tuple)

# Data operator
class Operator:
    """Data operator (parent class).

    Attributes:
        id (string): Unique operator ID.
        name (string): Operator name.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    def __init__(self, id=None, name=None, track_prov=False,
                                           propagate_prov=False):
        self.id = _generate_uuid() if id is None else id
        self.name = "Undefined" if name is None else name
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        logger.debug("Created {} operator with id {}".format(self.name,
                                                             self.id))

    # NOTE (john): Must be implemented by the subclasses
    def get_next(self):
        logger.error("Method not implemented!")

    # NOTE (john): Must be implemented by the subclasses
    def lineage(self, tuples: List[ATuple]) -> List[List[ATuple]]:
        logger.error("Lineage method not implemented!")

    # NOTE (john): Must be implemented by the subclasses
    def where(self, att_index: int, tuples: List[ATuple]) -> List[List[Tuple]]:
        logger.error("Where-provenance method not implemented!")

# Scan operator
class Scan(Operator):
    """Scan operator.

    Attributes:
        filepath (string): The path to the input file.
        filter (function): An optional user-defined filter.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes scan operator
    def __init__(self, filepath,bsize,filter=None, track_prov=False,
                                              propagate_prov=False):
        super(Scan, self).__init__(name="Scan", track_prov=track_prov,
                                   propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.filepath = filepath
        self.reader = self.readf(self.filepath,bsize)
        self.totaltuple = []
        self.identifier =''
        pass



    def readf(self,filepath,bsize):

        with open(filepath) as file:

            freader =csv.reader(file, delimiter=' ')
            batch_tuple=[];
            lineCount = 0

            for line in freader:
                temptuple = tuple([int(val) for val in line])
                lineCount+=1
                if self.propagate_prov:
                    identifer = str(filepath)[0]
                    metastring = identifer + str(lineCount)
                    Mytuple = ATuple(tuple=temptuple,metadata= metastring ,operator=self)
                else:
                    Mytuple = ATuple(tuple=temptuple, operator=self)

                batch_tuple.append(Mytuple)

                self.totaltuple.append(Mytuple)
                if len(batch_tuple)==bsize:
                    yield batch_tuple
                    batch_tuple = []
                # Returns next batch of tuples in given file (or None if file exhausted)
    def get_next(self):
        try:
            return next(self.reader)
        except StopIteration:
            return None
        # YOUR CODE HERE
        pass

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    # def where(self, att_index, tuples):
    def where(self, att_index, tuples):
        resultlist = []
        count = 0
        for i in tuples:
            result = []
            # Find the line of tuple in the file
            for scanTuple in self.totaltuple:
                count += 1
                if i == scanTuple:
                    result = [self.filepath,count,scanTuple,scanTuple.tuple[2]] # produce the result tuple of Where-provenance query
                    resultlist.extend(result)
        return resultlist
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

# Equi-join operator


class Join(Operator):
    """Equi-join operator.

    Attributes:
        left_input (Operator): A handle to the left input.
        right_input (Operator): A handle to the left input.
        left_join_attribute (int): The index of the left join attribute.
        right_join_attribute (int): The index of the right join attribute.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes join operator
    def __init__(self, left_input, right_input, left_join_attribute,
                                                right_join_attribute,
                                                track_prov=False,
                                                propagate_prov=False):
        super(Join, self).__init__(name="Join", track_prov=track_prov,
                                   propagate_prov=propagate_prov)
        self.left_input = left_input
        self.right_input = right_input
        self.left_join_attribute = left_join_attribute
        self.right_join_attribute = right_join_attribute
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.allSelectedLeft = []
        if self.track_prov:
            self.input_to_output = []

        # YOUR CODE HERE
        pass

    # Returns next batch of joined tuples (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        allLeftTuple=[]
        while True:
            tempLeft = self.left_input.get_next()

            if tempLeft == None:
                break
            else:
                allLeftTuple.extend(tempLeft)
                self.allSelectedLeft = allLeftTuple


        left ={}
        joined_tuple_list = []
        for tuple_k in self.allSelectedLeft:
            left.setdefault(tuple_k.tuple[self.left_join_attribute],tuple_k)

        right_batch = self.right_input.get_next()
        if right_batch == None:
            return None

        for tuple_i in right_batch:
            key = tuple_i.tuple[self.right_join_attribute]
            matched_tuple = left.get(key)
            if(matched_tuple!=None):

                left_tuple = matched_tuple
                right_tuple = tuple_i

                temp_tuple=[]

                for i in matched_tuple.tuple:

                    temp_tuple.append(i)

                for k in tuple_i.tuple:

                    temp_tuple.append(k)


                if self.propagate_prov:
                    metastring = left_tuple.metadata+"*"+right_tuple.metadata
                    joinedTuple = ATuple(tuple(temp_tuple),metadata=metastring, operator=self)
                else:
                    joinedTuple = ATuple(tuple(temp_tuple), operator=self)


                if self.track_prov:
                    inToOut = [joinedTuple,left_tuple,right_tuple]
                    self.input_to_output.append(inToOut)

                joined_tuple_list.append(joinedTuple)

        return joined_tuple_list
        # return left
        pass

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        resultlist = []
        for i in tuples:
            for left_right in self.input_to_output:
                if i == left_right[0]:
                    result = [left_right[1],left_right[2]]
                    resultlist.extend(result)
        return resultlist



        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    # def where(self, att_index, tuples):
    def where(self, att_index, tuples):
        resultlist = []
        for i in tuples:
            for k in self.input_to_output:
                if i == k[0]:
                    result = k[2].where(att_index)
                    resultlist.extend(result)


        return resultlist
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

# Project operator
class Project(Operator):
    """Project operator.

    Attributes:
        input (Operator): A handle to the input.
        fields_to_keep (List(int)): A list of attribute indices to keep.
        If empty, the project operator behaves like an identity map, i.e., it
        produces and output that is identical to its input.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes project operator
    def __init__(self, input, fields_to_keep=[], track_prov=False,
                                                 propagate_prov=False):
        super(Project, self).__init__(name="Project", track_prov=track_prov,
                                      propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.input = input
        self.fields_to_keep = fields_to_keep
        self.key = None
        if self.track_prov:
            self.input_to_output = []
        pass

    # Return next batch of projected tuples (or None if done)
    def get_next(self):
        result = []
        joined_tuple = self.input.get_next()
        if joined_tuple == None:
            return None


        if joined_tuple != []:
            for i in joined_tuple:

                self.key = i.tuple[3]
                tempT = []
                for k in self.fields_to_keep:
                    tempT.append(i.tuple[k])

                if self.propagate_prov:
                    resultTuple = ATuple(tuple(tempT),metadata=i.metadata,operator=self)

                else:
                    resultTuple = ATuple(tuple(tempT), operator=self)

                result.append(resultTuple)

                if self.track_prov:
                    inToOut = [resultTuple,joined_tuple]
                    self.input_to_output.append(inToOut)

        return result


        # YOUR CODE HERE
        pass

    # Returns the lineage of the given tuples
    def lineage(self, tuples):

        resultlist = []
        for p in tuples:
            for joinedTuple in self.input_to_output:
                if p == joinedTuple[0]:
                    result = joinedTuple[1][0].lineage()
                    resultlist.extend(result)
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        return resultlist
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):

        resultlist = []
        for i in tuples:
            for k in self.input_to_output:
                if i == k[0]:
                    result = k[1][0].where(att_index=att_index)
                    resultlist.extend(result)
        return resultlist
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

# Group-by operator
class GroupBy(Operator):
    """Group-by operator.

    Attributes:
        input (Operator): A handle to the input
        key (int): The index of the key to group tuples.
        value (int): The index of the attribute we want to aggregate.
        agg_fun (function): The aggregation function (e.g. AVG)
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes average operator
    def __init__(self, input, key=None, value=None, agg_fun=None, track_prov=False,
                                                   propagate_prov=False):
        super().__init__(name="GroupBy", track_prov=track_prov,
                                      propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.input = input #project op
        self.agg_fun = agg_fun
        self.state = None
        self.keylist =[]
        self.key = key #list of movieid
        self.value = value
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.keycount = 0
        self.keylen = 0
        self.dictlen = 0
        self.allProjectTuple =[]
        if self.track_prov:
            self.input_to_output = []


        pass
    # Returns aggregated value per distinct key in the input (or None if done)
    def get_next(self):

        while True:
            projectTuple = projectOp.get_next()
            if projectTuple == None:
                break
            self.allProjectTuple.extend(projectTuple)
        if self.state == None:
            self.state={}
            for i in self.allProjectTuple:
                self.state.setdefault(i.tuple[1], []).append(i)



        self.keylist = list(self.state.keys())
        self.keylen =len(self.state)

        if self.keycount<self.keylen:
            key = self.keylist[self.keycount]
            self.keycount+=1
            value = self.state.get(key)
            agg_value = self.agg_fun(value)
            temp_t = [key, agg_value]

            if self.propagate_prov:
                metalist = []
                metalisttest = []
                for i in value:

                    metastring = "("+i.metadata+"@"+str(i.tuple[2])+")"
                    metalist.append(metastring)

                resultTuple = ATuple(tuple=tuple(temp_t),metadata=metalist, operator=self)
            else:
                resultTuple = ATuple(tuple=tuple(temp_t), operator=self)


            if self.track_prov:
                inToOut = [resultTuple, value]
                self.input_to_output.append(inToOut)

            return resultTuple
        else:
            return None



    # Returns the lineage of the given tuples
    def lineage(self, tuples):

        resultlist = []
        for g in tuples:
            input_value_list = self.state.get(g.tuple[0])
            for i in input_value_list:
                result = i.lineage()
                resultlist.extend(result)
        return resultlist

        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

# Custom histogram operator
class Histogram(Operator):
    """Histogram operator.

    Attributes:
        input (Operator): A handle to the input
        key (int): The index of the key to group tuples. The operator outputs
        the total number of tuples per distinct key.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes histogram operator
    def __init__(self, input, key=0, track_prov=False, propagate_prov=False):
        super(Histogram, self).__init__(name="Histogram",
                                        track_prov=track_prov,
                                        propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.input = input
        self.key = key
        pass

    # Returns histogram (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        pass

# Order by operator
class OrderBy(Operator):
    """OrderBy operator.

    Attributes:
        input (Operator): A handle to the input
        comparator (function): The user-defined comparator used for sorting the
        input tuples.
        ASC (bool): True if sorting in ascending order, False otherwise.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes order-by operator
    def __init__(self, input, comparator=None, ASC=True, track_prov=False,
                                                    propagate_prov=False):
        super(OrderBy, self).__init__(name="OrderBy",
                                      track_prov=track_prov,
                                      propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.input = input
        self.comparator = comparator
        self.ASC = ASC
        if self.track_prov:
            self.input_to_output = []

        pass

    # Returns the sorted input (or None if done)
    def get_next(self):
        tuplelist = []
        while True:
            groupTuple = self.input.get_next()
            if groupTuple == None:
                break
            else:
                tuplelist.append(groupTuple)
                if self.track_prov:
                    self.input_to_output.append(groupTuple)

        tuplelist = self.comparator(tuplelist,self.ASC)
        result = []
        for row in tuplelist:
            if self.propagate_prov:
                tmptuple = ATuple(tuple=row.tuple,metadata=row.metadata, operator=self)
            else:
                tmptuple = ATuple(tuple=row.tuple, operator=self)
            result.append(tmptuple)
        return result

        # return intList
        # YOUR CODE HERE
        pass

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # resultlist = []
        for r in tuples:
            for i in self.input_to_output:
                if r.tuple == i.tuple:
                    result = i.lineage()
                    return result

        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

# Top-k operator
class TopK(Operator):
    """TopK operator.

    Attributes:
        input (Operator): A handle to the input.
        k (int): The maximum number of tuples to output.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes top-k operator
    def __init__(self, input, k=None, track_prov=False, propagate_prov=False):
        super(TopK, self).__init__(name="TopK", track_prov=track_prov,
                                   propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.input = input
        self.k = k
        pass

    # Returns the first k tuples in the input (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        result = []
        orderedTuple = self.input.get_next()
        for index,element in enumerate(orderedTuple):
            if index==self.k:
                break
            else:
                result.append(element)

        return result
        pass

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

# Filter operator
class Select(Operator):
    """Select operator.

    Attributes:
        input (Operator): A handle to the input.
        predicate (function): The selection predicate.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes select operator
    def __init__(self, input, predicate, track_prov=False,
                                         propagate_prov=False):
        super().__init__(name="Select", track_prov=track_prov,
                                     propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.predicate = predicate
        self.input = input
        if self.track_prov:
            self.input_to_output = []
        pass

    # Returns next batch of tuples that pass the filter (or None if done)
    def get_next(self):
        scantuple = self.input.get_next()
        filtered_tuple = []
        if scantuple == None:
            return None
        for tuple in scantuple:
            if(self.predicate.apply(tuple.tuple)):
                filtered_tuple.append(tuple)

        return filtered_tuple
        # YOUR CODE HERE
        pass

class Filter:
    def __init__(self,id,index):
        self.id = id
        self.att = index
    def apply(self,tuple):
        return tuple[self.att] == self.id










if __name__ == "__main__":

    logger.info("Assignment #1 Task1")



    # TASK 1: Implement 'likeness' prediction query for User A and Movie M
    #
    # SELECT AVG(R.Rating)
    # FROM Friends as F, Ratings as R
    # WHERE F.UID2 = R.UID
    #       AND F.UID1 = 'A'
    #       AND R.MID = 'M'
    # YOUR CODE HERE
    #

    def AVG(allProjectTuple):
        total_rating = 0
        for t in allProjectTuple:
            total_rating += t.tuple[2]
        if len(allProjectTuple) == 0:
            return 0
        else:
            avg = total_rating/len(allProjectTuple)
            return round(avg,2)




    # Avg test
    scanFriendOp = Scan(toyfriend, 1)
    friendFilter = Filter(1190 , 0)
    friendSelectOp = Select(scanFriendOp, friendFilter)
    scanMovieOP = Scan(toymovie, 1)
    movieFilter = Filter(0, 1)
    movieSelectOp = Select(scanMovieOP, movieFilter)
    joinOp = Join(friendSelectOp, movieSelectOp, left_join_attribute=1, right_join_attribute=0)
    projectOp = Project(joinOp, [1, 3, 4])
    #
    allProjectTuple = []
    while True:
        projectTuple = projectOp.get_next()
        if projectTuple == None:
            break
        allProjectTuple.extend(projectTuple)


    print(allProjectTuple)
    rate = AVG(allProjectTuple)
    print(rate)




    logger.info("Assignment #1 Task2")

    # TASK 2: Implement recommendation query for User A
    #
    # SELECT R.MID
    # FROM ( SELECT R.MID, AVG(R.Rating) as score
    #        FROM Friends as F, Ratings as R
    #        WHERE F.UID2 = R.UID
    #              AND F.UID1 = 'A'
    #        GROUP BY R.MID
    #        ORDER BY score DESC
    #        LIMIT 1 )

    def customComp(tlist, flag):
        if flag:
            tlist = sorted(tlist, key=cmp_to_key(lambda x, y: x.tuple[1] - y.tuple[1]))
        else:
            tlist = sorted(tlist, key=cmp_to_key(lambda x, y: y.tuple[1] - x.tuple[1]))

        return tlist

    
    scanFriendOp = Scan(toyfriend, 1)
    friendFilter = Filter(1190, 0)
    friendSelectOp = Select(scanFriendOp, friendFilter)
    scanMovieOp = Scan(toymovie, 1)
    joinOp = Join(friendSelectOp, scanMovieOp, left_join_attribute=1, right_join_attribute=0)
    projectOp = Project(joinOp, [1, 3, 4])
    groupOp = GroupBy(input=projectOp,agg_fun=AVG)
    orderByOp = OrderBy(input=groupOp,comparator=customComp,ASC=False)
    topKOp = TopK(orderByOp,1)
    result = topKOp.get_next()
    print(result)



    # YOUR CODE HERE


    # TASK 3: Implement explanation query for User A and Movie M
    #
    # SELECT HIST(R.Rating) as explanation
    # FROM Friends as F, Ratings as R
    # WHERE F.UID2 = R.UID
    #       AND F.UID1 = 'A'
    #       AND R.MID = 'M'

    # Hist_rating = Histogram(1190,0).get_next()
    # for i,element in enumerate(Hist_rating):
    #     print('{}{}{}'.format("Rating",i+1,element))
    # print(Hist_rating)

    # YOUR CODE HERE


    # TASK 4: Turn your data operators into Ray actors
    #
    # NOTE (john): Add your changes for Task 4 to a new git branch 'ray'


    logger.info("Assignment #2")

    # TASK 1: Implement lineage query for movie recommendation

    # YOUR CODE HERE

    logger.info("Assignment #2 Task1")
    scanFriendOp = Scan(toyfriend, 1)
    friendFilter = Filter(1190, 0)
    friendSelectOp = Select(scanFriendOp, friendFilter)
    scanMovieOp = Scan(toymovie, 1)
    joinOp = Join(friendSelectOp, scanMovieOp, left_join_attribute=1, right_join_attribute=0,track_prov=True)
    projectOp = Project(joinOp, [1, 3, 4],track_prov=True)
    groupOp = GroupBy(input=projectOp, agg_fun=AVG,track_prov=True)
    orderByOp = OrderBy(input=groupOp, comparator=customComp, ASC=False, track_prov=True)
    topKOp = TopK(orderByOp, 1)
    result = topKOp.get_next()
    print(result[0])
    print(result[0].lineage())


    # # TASK 2: Implement where-provenance query for 'likeness' prediction

    # # YOUR CODE HERE
    logger.info("Assignment #1 Task2")
    scanFriendOp = Scan(toyfriend, 1)
    friendFilter = Filter(1190, 0)
    friendSelectOp = Select(scanFriendOp, friendFilter)
    scanMovieOp = Scan(toymovie, 1)
    movieFilter = Filter(1, 1)
    movieSelectOp = Select(scanMovieOp, movieFilter)
    joinOp = Join(friendSelectOp, movieSelectOp, left_join_attribute=1, right_join_attribute=0,track_prov=True)
    projectOp = Project(joinOp, [1, 3, 4],track_prov=True)
    allProjectTuple = []
    while True:
        projectTuple = projectOp.get_next()
        if projectTuple == None:
            break
        allProjectTuple.extend(projectTuple)
    
    wherelist = []
    for i in allProjectTuple:
        result = i.where(4)
        wherelist.append(result)
    print(wherelist)



    # TASK 3: Implement how-provenance query for movie recommendation

    # YOUR CODE HERE
    logger.info("Assignment #2 Task3")
    scanFriendOp = Scan(toyfriend, 1,propagate_prov=True)
    friendFilter = Filter(1190, 0)
    friendSelectOp = Select(scanFriendOp, friendFilter)
    scanMovieOp = Scan(toymovie, 1,propagate_prov=True)
    joinOp = Join(friendSelectOp, scanMovieOp, left_join_attribute=1, right_join_attribute=0,track_prov=True,propagate_prov=True)
    projectOp = Project(joinOp, [1, 3, 4] , track_prov=True,propagate_prov=True)
    groupOp = GroupBy(input=projectOp, agg_fun=AVG,track_prov=True, propagate_prov=True)
    orderByOp = OrderBy(input=groupOp, comparator=customComp, ASC=False, track_prov=True , propagate_prov=True)
    topKOp = TopK(orderByOp, 1)
    result = topKOp.get_next()
    print(result[0].how())


    # TASK 4: Retrieve most responsible tuples for movie recommendation

    # YOUR CODE HERE
    # scanFriendOp = Scan(friend_res, 1,propagate_prov=True)
    # friendFilter = Filter(0,0) #friend id 0
    # friendSelectOp = Select(scanFriendOp, friendFilter)
    #
    # scanMovieOp = Scan(movie_res, 1,propagate_prov=True)
    # joinOp = Join(friendSelectOp, scanMovieOp, left_join_attribute=1, right_join_attribute=0,track_prov=True,propagate_prov=True)
    # projectOp = Project(joinOp, [1, 3, 4] , track_prov=True,propagate_prov=True)
    # groupOp = GroupBy(input=projectOp, agg_fun=AVG,track_prov=True, propagate_prov=True)
    # orderByOp = OrderBy(input=groupOp, comparator=customComp, ASC=False, track_prov=True , propagate_prov=True)
    # topKOp = TopK(orderByOp, 1)
    # result = topKOp.get_next()
    # logger.info("task 3: result: "+str(result[0]))
    # # logger.info("task 3: result: " + result[0].tuple)
    # print(result[0].how())
    # print(result[0].responsible_inputs())
    # print(result[0].how())
    # print( isinstance(result[0].operator,OrderBy))






