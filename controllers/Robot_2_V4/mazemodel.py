import math

class maze():
    def __init__(self,minx=-0.5,maxx=0.5,miny=-0.5,maxy=0.5,gridwidthx=0.125,gridwidthy=0.125):
        self.gridwidthx = gridwidthx
        self.gridwidthy = gridwidthy

        self.minx = minx - self.gridwidthx
        self.maxx = maxx + self.gridwidthx
        self.miny = miny - self.gridwidthy
        self.maxy = maxy + self.gridwidthy

        self.totx = (self.maxx-self.minx)/self.gridwidthx
        self.toty = (self.maxy-self.miny)/self.gridwidthy
                
        
    def find_cell(self,gps):
        
        x = gps[0]
        y = gps[1]
        
        xgrid = math.floor((x-self.minx)/self.gridwidthx)
        ygrid = math.floor((y-self.miny)/self.gridwidthy)
        
        gridnum = xgrid*self.toty + ygrid
        gridnum = int(gridnum)
        
        return (xgrid,ygrid),gridnum
        
    def find_center(self,cellnum):
        
        xgrid = math.floor(cellnum / self.toty)
        ygrid = cellnum % self.toty
        
        x = self.minx + (xgrid + 0.5) * self.gridwidthx
        y = self.miny + (ygrid + 0.5) * self.gridwidthy
        
        return (x,y)
        
    def calc_target(self, dir, cellnum):
        
        '''
    
        Direction Guide:
        0 - East
        1 - West
        2 - North
        3 - South
        
        ''' 
        if dir == 0:
            target_cell = cellnum - 1
        elif dir == 1:
            target_cell = cellnum + 1
        elif dir == 2:
            target_cell = cellnum + self.toty
        elif dir == 3:
            target_cell = cellnum - self.toty
            
        target_cell = int(target_cell)
            
        return target_cell        
        
        
        
        
        
        
        
        
        
        
        
        