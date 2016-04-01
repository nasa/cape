
.. automodule:: cape.dataBook
    :members: get_xlim, get_ylim

    Global data book container class
    --------------------------------
    
        .. autoclass:: cape.dataBook.DataBook
            :members: InitDBComp,
                PlotCoeff,
                MatchTrajectory, UpdateTrajectory,
                GetDBMatch, GetTargetMatch, GetTargetMatches,
                GetTargetByName, ReadTarget,
                Sort, Write
    
    Individual data books
    ---------------------
                
        .. autoclass:: cape.dataBook.DBBase
            :members: ProcessColumns, Read, 
                EstimateLineCount, ProcessConverters,
                ArgSort, Sort, Write, GetTrajectoryIndex, FindMatch,
                PlotCoeffBase, PlotCoeff
                
        .. autoclass:: cape.dataBook.DBComp
            :members: PlotCoeff
            
        .. autoclass:: cape.dataBook.DBTarget
            :members: ReadData, ProcessColumns,
                ReadAllData, ReadDataByColumn,
                CheckColumn, UpdateTrajectory,
                PlotCoeff
                
    Data book classes for individual cases
    --------------------------------------
    
        .. autoclass:: cape.dataBook.CaseData
            :members: GetIterationIndex, ExtractValue,
                PlotValue, PlotValueHist
            
        .. autoclass:: cape.dataBook.CaseFM
            :members: AddData, TransformFM, ShiftMRP,
                GetStatsN, GetStats,
                PlotCoeff, PlotCoeffHist
                
        .. autoclass:: cape.dataBook.CaseResid
            :members: GetNOrders, GetNOrdersUnsteady, GetIterationIndex,
                PlotResid, PlotL1, PlotL2, PlotLInf
            
            
    Other :mod:`cape.dataBook` methods
    ----------------------------------